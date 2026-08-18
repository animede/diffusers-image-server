# -*- coding: utf-8 -*-
"""
T2I(QwenImagePipeline)/ I2I(QwenImageImg2ImgPipeline)のロードと Lightning LoRA 制御。

T2I モデルは2種類("qwen-image" 無印 / "2512" = Qwen/Qwen-Image-2512)で、どちらも
同じ t2i_group に排他的にロードされる(48GBでは両方常駐させない)。

2026-07-18変更: 旧「2512-4bit」(ovedrive/Qwen-Image-2512-4bit、bnb NF4 の自己完結
パイプライン + PEFT adapter 方式)を廃止し、公式 Qwen/Qwen-Image-2512 bf16 transformer に
2512用 Lightning LoRA を自前 fuse -> fp8 layerwise cast する方式(無印の fp8-lightning
分岐と同一レシピ)へ置き換えた。これに伴い 2512 も無印と同じ共有コンポーネント
(state.shared)+ transformer 差し替え構造になり、I2I(QwenImageImg2ImgPipeline)にも
対応した(旧NF4は Img2Img クラスを持たない自己完結パイプラインのため T2I 専用だった)。
共有コンポーネントの互換性確認結果は paths.py の T2I_2512_REPO コメント参照
(text_encoder/vae/tokenizer は sha256 完全一致、scheduler 設定も全キー同値)。

2026-08-17追加(CLAUDE.md 66番): 30番で導入した「2512 は fp8-lightning fuse方式のみ」
制限を解除し、GGUF量子化(unsloth/Qwen-Image-2512-GGUF)にも対応した。理由は48GB未満
(24GB/20GB級)のカードでも2512を使えるようにするため。GGUF量子化 transformer への
Lightning LoRA適用は、旧来の `pipe.load_lora_weights()` 経路(PEFTがGGUFLinearを
ターゲットモジュールとして認識できず失敗する、旧11番)ではなく、Edit(59番・62番)と
同じ transformer-level `load_lora_adapter()` 方式に統一した(無印T2IのGGUF経路にも
同時に適用される。旧 `_apply_t2i_loras()` は非GGUF・非fp8-lightning経路(通常bf16)
専用に縮小した)。VRAM段階構成は Edit の3段階(59番)にならい、GGUF T2I/I2I にも
簡易版(full/te_fp8の2段階。model_cpu層は本タスクでは未実装、README参照)を追加した。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_t2i_group_locked()(行810-926)
  - _apply_t2i_loras()(行943-970)
  - set_t2i_lightning() / _set_lightning_adapters()(行1114-1146)
  - get_t2i_pipeline() / get_i2i_pipeline()(行1508-1529)

ロジック・既定値・実測コメントは無変更(勝手な改良はしない方針)。GGUF関連は上記の
2026-08-17追加分のみ新規。
"""
import time

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageImg2ImgPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)

from core import gpu
from core.config import (
    DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB,
    DS_EDIT_GGUF_MID_TE_QUANT,
    DS_EDIT_GGUF_MID_VRAM_FREE_THRESHOLD_GB,
)
from core.loaders import (
    fuse_lightning_lora_and_cast_to_fp8,
    load_transformer_from_config,
    load_transformer_from_pretrained_streaming,
    load_transformer_gguf,
)
from core.optimize import (
    apply_attention_backend,
    apply_compile,
    configure_transformer_offload,
    suspended_whole_module_swap_offload,
)
from core import progress as progress_mod
from core.resolve import COMFYUI_MODELS_DIR, resolve_model_path

from families.qwen_image import state
from families.qwen_image.paths import (
    BASE_REPO,
    T2I_2512_GGUF_CONFIG_REPO,
    T2I_2512_GGUF_FILENAME_TEMPLATE,
    T2I_2512_GGUF_HF_REPO,
    T2I_2512_LORA_4STEP_HF_FILE,
    T2I_2512_LORA_4STEP_HF_REPO,
    T2I_2512_LORA_4STEP_PATH,
    T2I_2512_REPO,
    T2I_GGUF_CONFIG_REPO,
    T2I_GGUF_FILENAME_TEMPLATE,
    T2I_GGUF_HF_REPO,
    T2I_LORA_4STEP_HF_FILE,
    T2I_LORA_4STEP_HF_REPO,
    T2I_LORA_4STEP_PATH,
    T2I_LORA_8STEP_HF_FILE,
    T2I_LORA_8STEP_HF_REPO,
    T2I_LORA_8STEP_PATH,
    T2I_TRANSFORMER_HF_FILE,
    T2I_TRANSFORMER_HF_REPO,
    T2I_TRANSFORMER_PATH,
    is_2512_model,
)
from families.qwen_image.runtime import is_fp8_lightning_quant, quant_suffix
from families.qwen_image.shared import load_shared_components_locked, warn_if_model_cpu_with_quant

import os


# ============================================================================
# T2I GGUF(24GB級)簡易 VRAM 段階構成(CLAUDE.md 59番の Edit 3段階を踏襲、2026-08-17追加)
# ============================================================================
# Edit の3段階(full / gguf_te_fp8 / model_cpu、families/qwen_image/edit.py)のうち、
# T2I/I2I では full(48GB級、transformer+共有ともフル常駐)と gguf_te_fp8(24GB級、
# 共有text_encoderのみ量子化)の2段階のみ実装する(model_cpu 層は本タスクでは未実装。
# 20GB級カードでの動作は「未検証」として README に記録する)。閾値は Edit と同じ
# 環境変数(DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB / DS_EDIT_GGUF_MID_VRAM_FREE_
# THRESHOLD_GB)を再利用する(T2I専用の環境変数は増やさない方針、CLAUDE.md 全体の
# 既存方針に合わせる)。
T2I_GGUF_TIER_FULL = "none"
T2I_GGUF_TIER_TE_FP8 = "gguf_te_fp8"


def _resolve_t2i_gguf_offload_tier() -> str:
    """T2I GGUF transformer をどの段階でロードするかを決める(Edit の
    _resolve_gguf_offload_tier() の簡易版。model_cpu 層は無いため、DS_OFFLOAD で
    model_cpu 系が明示された場合は未対応として警告し auto(自動判定)にフォールバックする。
    """
    from core.optimize import _OFFLOAD_ALIASES  # noqa: PLC0415 (循環import回避のため遅延import)

    requested = state.get_runtime_config().offload.strip().lower().replace("-", "_")
    if requested and requested != "auto":
        normalized = _OFFLOAD_ALIASES.get(requested, requested)
        if normalized == "none":
            print(f"[families.qwen_image] T2I GGUF offload tier: DS_OFFLOAD={requested!r} -> full(明示指定)")
            return T2I_GGUF_TIER_FULL
        if normalized == T2I_GGUF_TIER_TE_FP8:
            print(f"[families.qwen_image] T2I GGUF offload tier: DS_OFFLOAD={requested!r} -> gguf_te_fp8(明示指定)")
            return T2I_GGUF_TIER_TE_FP8
        print(
            f"[families.qwen_image] warning: DS_OFFLOAD={requested!r} is not supported for GGUF T2I/I2I "
            f"(有効値: none/gguf_te_fp8。model_cpu層は未実装); falling back to auto"
        )

    free_gb = gpu.free_vram_gb()
    if free_gb >= DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB:
        tier = T2I_GGUF_TIER_FULL
    else:
        tier = T2I_GGUF_TIER_TE_FP8
    print(
        f"[families.qwen_image] T2I GGUF offload tier: free VRAM={free_gb:.1f}GB -> {tier} "
        f"(thresholds: full>={DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB:.1f}GB)"
    )
    return tier


def _resolve_t2i_gguf_mid_te_quant() -> str:
    """gguf_te_fp8ティアで共有text_encoderをどう量子化するかを決める(Edit の
    _resolve_gguf_mid_te_quant() と同じ環境変数 DS_EDIT_GGUF_MID_TE_QUANT を再利用する)。
    """
    value = (DS_EDIT_GGUF_MID_TE_QUANT or "nf4").strip().lower()
    if value not in ("nf4", "fp8"):
        print(
            f"[families.qwen_image] warning: DS_EDIT_GGUF_MID_TE_QUANT={value!r} は未対応です"
            f"(対応値: nf4 / fp8)。既定の nf4 にフォールバックします"
        )
        return "nf4"
    return value


def _apply_t2i_gguf_lora(transformer, is_2512: bool) -> None:
    """GGUF量子化 T2I transformer(GGUFLinear層)へ Lightning 4steps LoRA を適用する
    (Edit の _apply_gguf_edit_lora() と同じ方式。CLAUDE.md 59番・62番参照)。

    **必ず CPU 上(GPU へ載せる前)で適用すること**(fp8-lightning系・Edit GGUFと同じ
    順序の罠、CLAUDE.md 17番)。失敗時はエラーにせず警告ログを出し、
    lora_available=False としてフォールバックする(無印T2IのGGUF経路の既存挙動と同じ
    防御方針)。
    """
    t2i_group = state.t2i_group
    try:
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers
        from safetensors.torch import load_file

        if is_2512:
            lora_path = resolve_model_path(
                T2I_2512_LORA_4STEP_PATH, T2I_2512_LORA_4STEP_HF_REPO, T2I_2512_LORA_4STEP_HF_FILE
            )
        else:
            lora_path = resolve_model_path(T2I_LORA_4STEP_PATH, T2I_LORA_4STEP_HF_REPO, T2I_LORA_4STEP_HF_FILE)
        converted_sd = _convert_non_diffusers_qwen_lora_to_diffusers(load_file(lora_path))
        transformer.load_lora_adapter(converted_sd, adapter_name="lightning4", prefix="transformer")
        # 既定は無効化しておく(有効化/無効化のトグルは pipeline 構築後、
        # set_t2i_lightning() が pipe.set_adapters()/pipe.disable_lora() で行う)。
        transformer.disable_lora()
        t2i_group["lora_available"] = True
        print(
            f"[families.qwen_image] GGUF T2I transformer: Lightning LoRA applied "
            f"(transformer-level adapter, model={'2512' if is_2512 else 'qwen-image'})"
        )
    except Exception as e:  # noqa: BLE001
        t2i_group["lora_available"] = False
        t2i_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        print(
            f"[families.qwen_image] warning: GGUF T2I transformer への Lightning LoRA適用に失敗しました。"
            f"LoRAなしにフォールバックします(steps/cfgを通常品質側に自動調整): {type(e).__name__}: {e}"
        )


def _load_t2i_group_locked() -> None:
    """T2I transformer をロードし、T2I/I2I パイプラインを構築する(共有コンポーネントを再利用)。

    どのモデル/量子化でロードするかは runtime_config(t2i_model / quant)で決まる:
      - t2i_model="2512": DS_QUANT=gguf-* なら GGUF量子化(unsloth/Qwen-Image-2512-GGUF、
        2026-08-17追加、CLAUDE.md 66番)、それ以外(fp8-lightning・none等)は
        fp8-lightning fuse 方式(Qwen/Qwen-Image-2512 bf16 シャード -> 2512用
        Lightning 4steps V1.0 fuse -> fp8 layerwise cast)。2512の非量子化bf16直接
        ロード経路は用意していないため、DS_QUANT=none指定時は警告してfp8-lightningへ
        フォールバックする。
      - t2i_model="qwen-image": DS_QUANT に応じて GGUF / fp8-lightning / bf16 の3経路(従来どおり)。

    fp8-lightning 選択時は Edit と同じ理由でロード順序を変える: bf16フルtransformer
    (約38GB)を先にGPUへロードしてLightning LoRAをfuseする必要があるため、共有コンポーネント
    (text_encoder等)のロードはそれより後に回す(先にロードするとVRAMの空きが減り、
    fuse時の一時ピークが収まらなくなるリスクが高まるため)。GGUF/非量子化時は従来通り
    共有コンポーネントを先にロードする。

    抽出元: pipeline_manager.py _load_t2i_group_locked()(行810-926)。
    """
    t2i_group = state.t2i_group
    if t2i_group["loaded"]:
        return
    raw_quant = state.get_runtime_config().quant
    quant = quant_suffix(raw_quant)
    fp8_lightning = is_fp8_lightning_quant(raw_quant)
    is_2512 = is_2512_model(state.get_runtime_config().t2i_model)
    # 2026-08-17変更(CLAUDE.md 66番): 「2512はfp8-lightningのみ」制限を解除した。
    # GGUF(quant真)の場合はそのままGGUF経路へ進む(下のif quant分岐がis_2512を見て
    # リポジトリ/config/LoRAを差し替える)。GGUFでもfp8-lightningでもない場合
    # (DS_QUANT=none等)のみ、2512はfp8-lightningへの強制フォールバックを維持する
    # (2512の非量子化bf16直接ロード経路は用意していないため)。
    if is_2512 and not quant and not fp8_lightning:
        print(
            f"[families.qwen_image] warning: t2i_model=2512 は fp8-lightning または GGUF 方式のみ対応です"
            f"(DS_QUANT={raw_quant!r} は無視して fp8-lightning でロードします)"
        )
        fp8_lightning = True
    small_transformer = bool(quant) or fp8_lightning

    # 2026-08-17変更(CLAUDE.md 66番): GGUF(quant真)は full/gguf_te_fp8 どちらのティアでも
    # 共有コンポーネントのロード可否・順序をこの関数内(quant分岐)で個別に制御する
    # (full ティアは transformer 常駐サイズが小さい(Q4_K_M実測約12.3GiB)ため従来どおり
    # 先に共有をロードしてよいが、te_fp8ティア+nf4選択時は逆順(TE先行)が必要なため)。
    # 非GGUF・非fp8-lightning(通常bf16)は従来どおりここで先にロードする。
    if not fp8_lightning and not quant:
        load_shared_components_locked(small_transformer_active=small_transformer)

    t0 = time.time()
    offload_mode = state.get_offload_mode(small_transformer_active=small_transformer)
    load_device = state.t2i_load_device(offload_mode)
    if not fp8_lightning and not quant:
        warn_if_model_cpu_with_quant(quant, offload_mode)

    gguf_tier = None  # T2I_GGUF_TIER_FULL / T2I_GGUF_TIER_TE_FP8(quant時のみ使う)
    gguf_mid_te_quant = None  # gguf_tier==T2I_GGUF_TIER_TE_FP8 の場合のみ "nf4"/"fp8"
    if quant:
        # GGUF量子化(2026-08-17拡張、CLAUDE.md 66番): 2512/無印いずれもGGUF対応。
        # 24GB級カード対応の簡易2段階(full/gguf_te_fp8)を Edit と同じ環境変数
        # (DS_EDIT_GGUF_*)で判定する。gguf_te_fp8ティア + nf4選択時は Edit の
        # ロード順序の罠(nf4ロードの一時ピーク13.83GiB)と同じ理由で、共有
        # text_encoderをtransformerより先にロードする。
        gguf_tier = _resolve_t2i_gguf_offload_tier()
        if gguf_tier == T2I_GGUF_TIER_FULL:
            # full ティア: transformer(Q4_K_M実測約12.3GiB)は小さいため、従来のGGUF
            # 挙動どおり共有コンポーネントを先にロードしてよい。
            load_shared_components_locked(small_transformer_active=small_transformer)
        if gguf_tier == T2I_GGUF_TIER_TE_FP8:
            gguf_mid_te_quant = _resolve_t2i_gguf_mid_te_quant()
            if gguf_mid_te_quant == "nf4" and not state.shared["loaded"]:
                runtime_config_early = state.get_runtime_config()
                original_te_quant_early = runtime_config_early.te_quant
                if original_te_quant_early in ("", "none", "off", "bf16"):
                    print(
                        "[families.qwen_image] T2I GGUF te_fp8 tier (nf4): 共有text_encoderを"
                        "transformerより先にnf4ロードします(ロード順序の罠、CLAUDE.md 59番参照)"
                    )
                    runtime_config_early.te_quant = "nf4"
                    try:
                        load_shared_components_locked(
                            small_transformer_active=small_transformer, force_offload_mode="group"
                        )
                    finally:
                        runtime_config_early.te_quant = original_te_quant_early

        if is_2512:
            gguf_hf_repo = T2I_2512_GGUF_HF_REPO
            gguf_filename_template = T2I_2512_GGUF_FILENAME_TEMPLATE
            gguf_config_repo = T2I_2512_GGUF_CONFIG_REPO
        else:
            gguf_hf_repo = T2I_GGUF_HF_REPO
            gguf_filename_template = T2I_GGUF_FILENAME_TEMPLATE
            gguf_config_repo = T2I_GGUF_CONFIG_REPO
        filename = gguf_filename_template.format(suffix=quant)
        local_path = os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", filename)
        path = resolve_model_path(local_path, gguf_hf_repo, filename)
        print(
            f"[families.qwen_image] loading T2I transformer as GGUF({quant}) from {path} "
            f"(model={'2512' if is_2512 else 'qwen-image'}, tier={gguf_tier})"
        )
        transformer = load_transformer_gguf(QwenImageTransformer2DModel, path, gguf_config_repo)

        # GGUF量子化transformer(GGUFLinear層)へのLightning LoRA適用は transformer-level
        # load_lora_adapter() を使う(旧来の pipe.load_lora_weights() 経路は GGUFLinear を
        # PEFTが認識できず失敗する。CLAUDE.md 59番・62番参照)。**必ずGPUへ載せる前
        # (CPU上)で適用すること**(順序の罠、CLAUDE.md 17番)。
        _apply_t2i_gguf_lora(transformer, is_2512)

        # GGUFは小さいためCPU RAM常駐を避け、常にフルGPU常駐にする(group系オフロードは
        # 使わない。model_cpu層は本タスクでは未実装)。
        transformer.to("cuda")
        transformer_offload_mode = "none"
    elif fp8_lightning:
        # bf16 transformer(無印 Qwen/Qwen-Image または Qwen/Qwen-Image-2512)を直接GPUへ
        # ロード(実測 約38GB)し、その場で対応する Lightning LoRA(4steps)をfuseしてから
        # enable_layerwise_casting でストレージをfp8化する(Edit(2511)と同一手法)。
        # 共有コンポーネント読み込み前にVRAMの空きを最大限確保するため、offload_modeに
        # 関わらず常に "cuda:0" へ直接ロードする(group系オフロードは使わない)。
        #
        # 2026-07-18バグ修正(T2I fp8-lightning 4steps の霧がかかったような品質崩壊の根本原因):
        # ComfyUI配布の qwen_image_fp8_e4m3fn.safetensors(T2I_TRANSFORMER_PATH)は、
        # 元々fp8_e4m3fn精度で量子化された重みを bf16 として保存し直したファイルであることが
        # 実機検証で判明した(load_safetensors_streaming(dtype=bf16)でロードした後、代表8層の
        # 重みを一つずつ .to(float8_e4m3fn).to(float32) で往復させたところ、全要素が完全に
        # ロスレスでfp8往復できた=100%の要素で元の値と誤差ゼロ。つまりbf16の仮数部下位ビットは
        # 全て0で、実質的な精度はfp8のまま)。Lightning LoRA(4steps)をfuseすると加わる補正量は
        # 実測で平均絶対値 約3e-5(fp8_e4m3fnの量子化誤差の平均絶対値ともほぼ同じ大きさ)しかなく、
        # fuse後に enable_layerwise_casting で再度fp8へキャストすると、LoRAが加えた補正がfp8の
        # 量子化グリッドに再度丸め込まれてほぼ消滅してしまう(delta/quant_err比 ≈ 1.0を実測)。
        # これが「30steps/cfg4(Lightningなし)は完璧なのに、fp8-lightning 4stepsだけ霧がかかる」
        # 症状の直接原因(Lightning LoRAのfuseが実質no-op化していた)。対照実験として adapter
        # モード(fuseせずLoRAをPEFT経由で加算、baseのみfp8ストレージ)は同一seed/prompt/shiftで
        # シャープな出力になることを確認済み(LoRAの寄与がbf16 compute_dtypeで守られるため)。
        #
        # 対策: fp8-lightning経路は HF Hub の bf16 シャード(真にフル精度)を直接ストリーミング
        # ロードし、そこにLoRAをfuseしてからfp8化する(ComfyUIの事前量子化fp8ファイルは使わない)。
        # T2I_TRANSFORMER_PATH/HF_REPO/HF_FILE(ComfyUIファイル)は GGUF不使用・非量子化時の
        # 通常ロード(下の else 分岐)では引き続き使う(そちらはfuse+再量子化をしないため無害)。
        #
        # 2512(Qwen/Qwen-Image-2512)も全く同じレシピを適用する(2026-07-18導入):
        # ベースrepoとLoRA(lightx2v/Qwen-Image-2512-Lightning 4steps V1.0 bf16)を
        # 差し替えるだけで、fuse手順・fp8化・VRAMピーク特性は無印と同一。
        if is_2512:
            repo = T2I_2512_REPO
            lora4 = (T2I_2512_LORA_4STEP_PATH, T2I_2512_LORA_4STEP_HF_REPO, T2I_2512_LORA_4STEP_HF_FILE)
        else:
            repo = BASE_REPO
            lora4 = (T2I_LORA_4STEP_PATH, T2I_LORA_4STEP_HF_REPO, T2I_LORA_4STEP_HF_FILE)
        print(
            f"[families.qwen_image] loading T2I transformer for fp8-lightning fuse from "
            f"{repo}/transformer (HF Hub bf16 shards, 直接cuda:0へストリーミング)"
        )
        transformer = load_transformer_from_pretrained_streaming(
            QwenImageTransformer2DModel, repo, "transformer", "cuda:0"
        )
        print("[families.qwen_image] fusing T2I Lightning LoRA and casting to fp8_e4m3fn storage...")
        fuse_lightning_lora_and_cast_to_fp8(transformer, *lora4)
        if torch.cuda.is_available():
            print(f"[families.qwen_image] T2I fp8-lightning transformer ready, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB")
        transformer_offload_mode = "none"
    else:
        path = resolve_model_path(T2I_TRANSFORMER_PATH, T2I_TRANSFORMER_HF_REPO, T2I_TRANSFORMER_HF_FILE)
        print(f"[families.qwen_image] loading T2I transformer from {path} (load_device={load_device})")
        transformer = load_transformer_from_config(
            QwenImageTransformer2DModel, BASE_REPO, "transformer", path, load_device
        )
        transformer_offload_mode = offload_mode

    if fp8_lightning:
        # transformer をVRAMに確保した後で共有コンポーネント(text_encoder等)をロードする
        # (上記の順序に関する注意を参照)。
        load_shared_components_locked(small_transformer_active=small_transformer)
        warn_if_model_cpu_with_quant("fp8-lightning", offload_mode)

    # GGUF te_fp8ティア(2026-08-17追加): 共有text_encoderの量子化・ロードタイミングは
    # gguf_mid_te_quant によって分岐する。nf4選択時は上の早期プリロードで既にロード済み
    # (state.shared["loaded"]==True のためここは自然に素通りする)。fp8明示選択時は
    # Editと同じ理由(post-hoc castingはCPU上で行うためGPU一時ピークが無い)で
    # transformerロード後のこのタイミングでロードする。
    runtime_config = state.get_runtime_config()
    original_te_quant = runtime_config.te_quant
    forced_te_quant = None
    if (
        gguf_tier == T2I_GGUF_TIER_TE_FP8
        and gguf_mid_te_quant == "fp8"
        and not state.shared["loaded"]
        and original_te_quant in ("", "none", "off", "bf16")
    ):
        print(
            "[families.qwen_image] T2I GGUF te_fp8 tier: 共有text_encoderをfp8化します"
            "(DS_EDIT_GGUF_MID_TE_QUANT=fp8明示指定のため一時的にfp8へ切替、24GB級カード対応)"
        )
        runtime_config.te_quant = "fp8"
        forced_te_quant = "fp8"
    if quant and not state.shared["loaded"]:
        try:
            # te_fp8ティア(fp8明示)は素朴にGPU常駐("group")、full/fp8-lightning以外の
            # GGUFではグローバルoffload_modeが偶々group_lowvram等になっていた場合の
            # accelerate hook衝突を避けるため強制する(Edit shared.py と同じ理由)。
            force_mode = "group" if gguf_tier == T2I_GGUF_TIER_TE_FP8 else None
            load_shared_components_locked(small_transformer_active=small_transformer, force_offload_mode=force_mode)
        finally:
            if forced_te_quant is not None:
                runtime_config.te_quant = original_te_quant
    elif forced_te_quant is not None:
        runtime_config.te_quant = original_te_quant

    # scheduler は常に無印 BASE_REPO の設定を使う(Qwen/Qwen-Image-2512 の
    # scheduler_config.json と全キー同値であることを確認済み。paths.py 参照)。
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")

    if transformer_offload_mode == "model_cpu":
        t2i_pipe = QwenImagePipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            transformer=transformer,
        )
        t2i_pipe.enable_model_cpu_offload()
    else:
        if not quant and not fp8_lightning:  # quant/fp8_lightning時は上ですでに .to("cuda") 済み
            configure_transformer_offload(transformer, transformer_offload_mode)
        t2i_pipe = QwenImagePipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            transformer=transformer,
        )

    apply_attention_backend(transformer, state.get_runtime_config().attention_backend, state.get_runtime_config().device)
    apply_compile(transformer, state.get_runtime_config().compile)

    # I2I パイプラインは同一インスタンスを共有するラッパー(二重ロードしない)。
    # 2512 も同じ transformer 共有構造のため、I2I はモデルを問わず利用できる。
    i2i_pipe = QwenImageImg2ImgPipeline(
        scheduler=scheduler,
        vae=state.shared["vae"],
        text_encoder=state.shared["text_encoder"],
        tokenizer=state.shared["tokenizer"],
        transformer=transformer,
    )

    quant_label = "fp8-lightning" if fp8_lightning else quant
    if fp8_lightning:
        # Lightning LoRAはすでに重みにfuse済み。追加のLoRAロードは不要で、常にLightning
        # 適用状態として扱う(無効化不可)。
        t2i_group["lora_available"] = True
        t2i_group["lightning_merged"] = True
    elif quant:
        # GGUFは transformer-level で既に _apply_t2i_gguf_lora() 済み(上記参照)。
        # そちらが t2i_group["lora_available"] と初期無効化(transformer.disable_lora())を
        # 済ませているため、ここでは pipeline-level API(set_t2i_lightning() が使う
        # pipe.set_adapters()/pipe.disable_lora())と状態を揃えるための保険のみ行う
        # (Editの _load_edit_group_locked() と同じパターン、CLAUDE.md 59番)。
        if t2i_group.get("lora_available"):
            try:
                t2i_pipe.disable_lora()
            except Exception as e:  # noqa: BLE001
                print(f"[families.qwen_image] warning: GGUF T2I LoRA の初期無効化に失敗しました: {e}")
    else:
        _apply_t2i_loras(t2i_pipe, gguf_quantized=False)

    t2i_group["transformer"] = transformer
    t2i_group["t2i_pipe"] = t2i_pipe
    t2i_group["i2i_pipe"] = i2i_pipe
    t2i_group["scheduler"] = scheduler
    t2i_group["loaded"] = True
    t2i_group["load_time_s"] = time.time() - t0
    t2i_group["quant"] = quant_label
    t2i_group["gguf_tier"] = gguf_tier  # None(非GGUF) / "none" / "gguf_te_fp8"(66番)
    t2i_group["gguf_mid_te_quant"] = gguf_mid_te_quant  # gguf_tier=="gguf_te_fp8" の場合のみ "nf4"/"fp8"
    print(
        f"[families.qwen_image] T2I/I2I group loaded in {t2i_group['load_time_s']:.1f}s "
        f"(model={'2512' if is_2512 else 'qwen-image'}, offload_mode={transformer_offload_mode}, "
        f"quant={quant_label}, gguf_tier={gguf_tier}, gguf_mid_te_quant={gguf_mid_te_quant})"
    )


def _apply_t2i_loras(pipe, gguf_quantized: bool = False) -> None:
    """T2I/I2I 用 Lightning LoRA(4step / 8step)を両方ロードし、既定は無効化しておく。
    有効化は set_t2i_lightning() でリクエストごとに切り替える。

    この関数は無印 qwen-image の非量子化bf16経路(DS_QUANT=none)専用。2512 は常に
    fp8-lightning fuse のためここには来ない。GGUF量子化(2026-08-17、CLAUDE.md 66番)も
    ここには来なくなった: 旧実装は `pipe.load_lora_weights()` を使っていたが、これは
    GGUFLinear をPEFTのターゲットモジュール検出が認識できず
    "Target modules {...} not found" で失敗する(旧11番の制約、実機検証済み)。GGUF経路は
    代わりに transformer-level の `load_lora_adapter()` を使う _apply_t2i_gguf_lora()
    (呼び出し元 _load_t2i_group_locked() の quant 分岐参照)に移行した。gguf_quantized
    引数は後方互換のため残すが、現在の呼び出し元は常に False で呼ぶ。

    抽出元: pipeline_manager.py _apply_t2i_loras()(行943-970)。
    """
    t2i_group = state.t2i_group
    try:
        path4 = resolve_model_path(T2I_LORA_4STEP_PATH, T2I_LORA_4STEP_HF_REPO, T2I_LORA_4STEP_HF_FILE)
        path8 = resolve_model_path(T2I_LORA_8STEP_PATH, T2I_LORA_8STEP_HF_REPO, T2I_LORA_8STEP_HF_FILE)
        # group_lowvram の text_encoder 手動スワップ(accelerate の CpuOffload フック)が
        # 付いたまま load_lora_weights() を呼ぶと、diffusers がパイプライン全体を
        # model-cpu-offload 済みと誤検出してフックを剥がし、代わりに
        # enable_model_cpu_offload() を掛け直してしまう(group offload と衝突する)。
        # 詳細は core.optimize.suspended_whole_module_swap_offload() のdocstring参照。
        with suspended_whole_module_swap_offload(pipe.text_encoder):
            pipe.load_lora_weights(path4, adapter_name="lightning4")
            pipe.load_lora_weights(path8, adapter_name="lightning8")
        pipe.disable_lora()  # 既定は無効
        t2i_group["lora_available"] = True
    except Exception as e:  # noqa: BLE001
        t2i_group["lora_available"] = False
        t2i_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        if gguf_quantized:
            print(
                f"[families.qwen_image] warning: GGUF量子化transformerへのT2I Lightning LoRA適用に失敗しました"
                f"(既知の制約: PEFTがGGUFLinearを未対応)。LoRAなしにフォールバックします: "
                f"{type(e).__name__}: {e}"
            )
        else:
            print(f"[families.qwen_image] warning: T2I Lightning LoRA のロードに失敗しました: {e}")


def _set_lightning_adapters(pipe, group_state: dict, steps: int) -> bool:
    """steps に応じて Lightning LoRA(4step/8step アダプタ)を有効/無効化する共通処理。
    4 または 8 のときのみ有効化。有効化できた場合 True を返す
    (LoRA自体が未ロード/非対応の場合は常に False)。

    2026-08-17追加(CLAUDE.md 66番): GGUF経路(_apply_t2i_gguf_lora())は "lightning4"
    アダプタのみロードする(8step版の2512用LoRAが存在しないため、無印/2512とも4stepのみ
    提供)。8steps要求時に "lightning8" が存在しない場合、num_inference_steps=8のまま
    4step蒸留LoRAを適用すると品質ミスマッチになるため、**静かに4stepへ差し替えたりは
    せず**呼び出し側(run_t2i()/run_i2i())の既存フォールバック(lightning_unavailable=True
    -> 30steps/cfg4.0)に委ねる(False を返す)。
    """
    if not group_state.get("lora_available"):
        return False
    try:
        transformer = getattr(pipe, "transformer", None)
        available_adapters = set(getattr(transformer, "peft_config", {}) or {}) if transformer is not None else set()
        if steps == 8 and "lightning8" not in available_adapters:
            # GGUF経路(8step版LoRAが存在しない)。8stepsのまま4step LoRAを当てる
            # ミスマッチを避けるため、非対応として扱う(呼び出し側が30steps/cfg4.0へ
            # フォールバックする)。
            return False
        if steps in (4, 8):
            # enable_lora() 必須(CLAUDE.md 62番のno-opバグ修正、edit.py set_edit_lightning()
            # と同じ理由): disable_lora() が立てる PEFT の _disable_adapters フラグを
            # set_adapters() はクリアしないため、先に enable_lora() を呼ばないと
            # LoRA が forward に効かない no-op になる。
            pipe.enable_lora()
        if steps == 4:
            pipe.set_adapters(["lightning4"], adapter_weights=[1.0])
            return True
        if steps == 8:
            pipe.set_adapters(["lightning8"], adapter_weights=[1.0])
            return True
        pipe.disable_lora()
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[families.qwen_image] warning: Lightning LoRA 切替に失敗しました: {e}")
        return False


def set_t2i_lightning(pipe, steps: int) -> bool:
    """T2I グループの Lightning LoRA を有効/無効化する。

    fp8-lightning(重みにfuse済み。2512 は常にこれ)の場合は常に True を返す(無効化不可)。

    抽出元: pipeline_manager.py set_t2i_lightning()(行1135-1145)。
    """
    group = state.t2i_group
    if group.get("lightning_merged"):
        return True
    return _set_lightning_adapters(pipe, group, steps)


def _switch_t2i_model_if_needed(model: "str | None") -> None:
    """リクエストで明示された T2I モデル("2512" | "qwen-image"。旧値 "2512-4bit" は
    後方互換で "2512" として扱う)に応じて、現在アクティブなモデル
    (state.get_runtime_config().t2i_model)を切り替える。

    タスク1(UIからのT2Iモデル切替)対応: 48GBでは2512と無印Qwen-Imageの transformer を
    両方常駐させない方針のため、要求モデルが現在のグループと異なる場合は、T2Iグループを
    先に unload してからランタイム設定(t2i_model)を更新する。生成はグローバルロック1本
    (core.gpu.generation_lock)で排他されるため、この関数が呼ばれる時点で他の生成は
    走っていない前提(app.py の _generate_or_409() がロック取得後に registry.load() 経由で呼ぶ)。

    model が None(未指定)の場合は何もしない(現行の DS_T2I_MODEL 既定のまま)。
    """
    if model is None:
        return
    requested_2512 = is_2512_model(model)
    current_2512 = is_2512_model(state.get_runtime_config().t2i_model)
    if requested_2512 == current_2512:
        return  # 既に要求モデルがアクティブ(まだ未ロードの可能性はあるが同一グループ)

    from families.qwen_image import lifecycle

    need_switch = False
    with state.lock:
        # 再チェック(ロック取得までの間に他スレッドが切り替えた可能性を考慮)。
        current_2512 = is_2512_model(state.get_runtime_config().t2i_model)
        if requested_2512 == current_2512:
            return
        if requested_2512:
            print("[families.qwen_image] switching T2I model: qwen-image -> 2512")
        else:
            print("[families.qwen_image] switching T2I model: 2512 -> qwen-image")
        state.get_runtime_config().t2i_model = "2512" if requested_2512 else "qwen-image"
        need_switch = True

    # lifecycle.unload() は内部で state.lock を取得するため、上の with ブロックの外側で呼ぶ
    # (非再入ロックのためネストできない。CLAUDE.md 14番の罠と同じ理由)。
    # target="t2i" は t2i_group と、その transformer を参照する ControlNet グループを解放する
    # (解放順序は lifecycle.unload() 参照)。
    if need_switch:
        lifecycle.unload("t2i")


def _switch_t2i_quant_if_needed(quant: "str | None") -> None:
    """リクエストで明示された quant(例: "gguf-q4_k_m" / "fp8-lightning" / "none")に
    応じて、現在アクティブな量子化方式(state.get_runtime_config().quant)を切り替える
    (CLAUDE.md 69番、24番の _switch_t2i_model_if_needed() と同じ思想のパターン)。

    quant が None(runtime.normalize_quant_request() で正規化済み、"default"/未指定を含む)
    の場合は何もしない(現行の DS_QUANT 既定のまま)。
    """
    if quant is None:
        return
    from families.qwen_image.runtime import gguf_quant_vram_ok

    current = state.get_runtime_config().quant
    if quant == current:
        return  # 既に要求quantがアクティブ(まだ未ロードの可能性はあるが同一グループ)
    if not gguf_quant_vram_ok(quant):
        from core import gpu
        from core.config import VRAM_FREE_THRESHOLD_GB

        raise ValueError(
            f"この量子化方式(fp8-lightning)には空きVRAMが不足しています"
            f"(空き{gpu.free_vram_gb():.1f}GB < 必要{VRAM_FREE_THRESHOLD_GB:.1f}GB)。"
            f"GGUF版(例: quant=\"gguf-q4_k_m\")を選択してください。"
        )

    from families.qwen_image import lifecycle

    need_switch = False
    with state.lock:
        # 再チェック(ロック取得までの間に他スレッドが切り替えた可能性を考慮)。
        current = state.get_runtime_config().quant
        if quant == current:
            return
        print(f"[families.qwen_image] switching T2I/I2I quant: {current!r} -> {quant!r}")
        state.get_runtime_config().quant = quant
        need_switch = True

    # lifecycle.unload() は内部で state.lock を取得するため、上の with ブロックの外側で呼ぶ
    # (非再入ロックのためネストできない、CLAUDE.md 14番と同じ理由)。
    if need_switch:
        lifecycle.unload("t2i")


def get_t2i_pipeline(model: "str | None" = None, quant: "str | None" = None) -> QwenImagePipeline:
    """DS_T2I_MODEL(または引数 model で明示指定)に応じて 2512 / 無印 Qwen-Image の
    いずれかを返す。model 指定時、現在アクティブなモデルと異なれば自動的に切り替える
    (T2Iグループをunloadしてから再ロードする。タスク1対応)。

    quant(CLAUDE.md 69番、2026-08-18追加): 明示指定時、現在アクティブな量子化方式と
    異なれば自動的に切り替える(bf16/fp8-lightning ⇔ GGUF)。

    抽出元: pipeline_manager.py get_t2i_pipeline()(行1508-1520)。
    """
    _switch_t2i_model_if_needed(model)
    _switch_t2i_quant_if_needed(quant)
    if not state.t2i_group["loaded"]:
        with state.lock:
            if not state.t2i_group["loaded"]:
                _load_t2i_group_locked()
    pipe = state.t2i_group["t2i_pipe"]
    progress_mod.disable_diffusers_tqdm(pipe)
    return pipe


def get_i2i_pipeline(model: "str | None" = None, quant: "str | None" = None) -> QwenImageImg2ImgPipeline:
    """I2I パイプラインを返す。2026-07-18から I2I も "2512" | "qwen-image" の両方に対応
    (2512 が無印と同じ transformer 共有構造になったため。QwenImageImg2ImgPipeline は
    _load_t2i_group_locked() が T2I と同一 transformer から構築済み)。

    旧実装との差分: 旧NF4(2512-4bit)は Img2Img クラスを持たない自己完結パイプライン
    だったため I2I 非対応で、model="2512-4bit" 指定時は ValueError(400)を返し、
    さらに t2i_2512_group 常駐時のOOM回避・t2i_model 同期の特殊処理を持っていた。
    新方式では T2I と同じ _switch_t2i_model_if_needed() による排他切替に一本化され、
    これらの特殊処理は不要になった(グループが1つになったため、モデル不整合による
    二重常駐OOM自体が構造的に起きない)。

    抽出元: pipeline_manager.py get_i2i_pipeline()(行1523-1529)。
    """
    _switch_t2i_model_if_needed(model)
    _switch_t2i_quant_if_needed(quant)
    if not state.t2i_group["loaded"]:
        with state.lock:
            if not state.t2i_group["loaded"]:
                _load_t2i_group_locked()
    pipe = state.t2i_group["i2i_pipe"]
    progress_mod.disable_diffusers_tqdm(pipe)
    return pipe
