# -*- coding: utf-8 -*-
"""
Edit(QwenImageEditPlusPipeline、Qwen-Image-Edit-2511)のロードと Lightning LoRA 制御。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_edit_group_locked()(行1200-1321)
  - _apply_edit_loras()(行1324-1349)
  - set_edit_lightning()(行1352-1368)
  - set_scheduler_shift()(行1371-1375)
  - get_edit_pipeline()(行1532-1537)

ロジック・既定値・実測コメント・ロード順序(transformer 先読み)は無変更。

2026-08-16追加(CLAUDE.md 59番参照): GGUF(DS_QUANT=gguf-*)経路に、空きVRAMに応じた
3段階構成を追加した(_resolve_gguf_offload_tier() が段階を決定し、
_load_edit_group_locked() 内の quant 分岐と _apply_gguf_edit_lora() がそれぞれの
ロード手順を実装する)。48GB級では従来どおり全常駐(オフロードなし)を完全に維持し、
24GB級はtransformer常駐+text_encoderのみ量子化(既定nf4、DS_EDIT_GGUF_MID_TE_QUANTで
fp8に切替可)、20GB級のみ enable_model_cpu_offload() を使う。DS_OFFLOAD で明示指定
された場合はユーザー指定を尊重する(auto_offload_mode の自動判定を経由しない)。

2026-08-16追加(24GB級 te_fp8ティアのnf4化): 24GB級の常駐ピークは fp8化TE(実測
8.74GiB)だと GGUF transformer(12.32GiB常駐)と合わせて実測23.27GB前後になり、実物の
24GBカード(空き~23.5GB)では境界事故(空き23.7GBでVAE decode中にOOM)を実機で確認した。
text_encoderを bnb nf4 量子化(常駐5.54GiB)に変えることでピークを ~18-20GB に下げる。
**nf4 のロードはbf16でmaterializeしてから量子化するため一時ピーク13.83GiBが発生する**。
そのためnf4選択時のみロード順序を「TE(nf4)を先にロード → GGUF transformerを後で
ロード」に反転する(transformerを先に確保していると13.83GiBの一時ピークが乗り24GB級
カードでOOMするため)。full/model_cpuティア、および te_fp8ティアでfp8を明示選択した
場合の順序(transformer先行)は変更していない。

--- 本サーバ独自の追加(charsheet の新設計、diffusers-server 本体の
edit_angles/edit_angles_bf16/edit_angles_bf16group 3変種システムは移植しない) ---

このリポジトリの charsheet は「Edit グループの transformer に fal 2511
Multiple-Angles LoRA(EDIT_LORA_ANGLES_*)を2本目の transformer-level アダプタ
として常時ロードしておき、リクエストごとに enable_lora()/set_adapters()/
disable_lora() でトグルする」設計にした(diffusers-server 本体の
scratchpad/probe_gguf_charsheet.py で実証済みの設計をそのまま採用)。

- GGUF経路: `transformer.load_lora_adapter(lightning_sd, adapter_name="lightning",
  prefix="transformer")` の直後に `transformer.load_lora_adapter(angles_sd,
  adapter_name="angles", prefix="transformer")` を追加で呼ぶ(いずれも
  `.to("cuda")` より **前**、CPU上で適用する。CLAUDE.md 17番・59番の順序の罠)。
- fp8-lightning経路: Lightning は既に重みへ fuse 済み(adapter ではない)なので、
  angles だけを2本目のアダプタとして追加する。`fuse_lightning_lora_and_cast_to_fp8()`
  でLightningをfuseした直後・`enable_layerwise_casting()` を呼ぶ **前**に
  `transformer.load_lora_adapter(angles_sd, adapter_name="angles",
  prefix="transformer")` を挟む(core.loaders.load_lora_adapters_and_cast_to_fp8()
  と同じ「LoRAアダプタは fp8 キャストより先にロードする」順序の鉄則。fp8化後に
  ロードすると新規作成される lora_A/lora_B が fp8 dtype で作られてしまい
  `NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'` になる)。
- トグル: 通常Editリクエストは `set_edit_lightning()`(Lightningのみ有効化、GGUFなら
  `set_adapters(["lightning"], ...)`、fp8-lightningならLightningは常時fuse済みなので
  何もしない)。charsheetリクエストは `set_edit_angles(pipe, True)` を追加で呼び、
  GGUFなら `set_adapters(["lightning", "angles"], weights=[1.0, 1.0])`、
  fp8-lightningなら `set_adapters(["angles"], weights=[1.0])`(Lightningは既にfuse
  済みの重みなので明示不要)にする。常に `enable_lora()` を `set_adapters()` より
  **先**に呼ぶこと(62/63番のno-opバグの再発防止)。
"""
import math
import time

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageImg2ImgPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import snapshot_download
from transformers import Qwen2VLProcessor

from core import gpu
from core.config import (
    DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB,
    DS_EDIT_GGUF_MID_TE_QUANT,
    DS_EDIT_GGUF_MID_VRAM_FREE_THRESHOLD_GB,
    DS_EDIT_GGUF_MODEL_CPU_MIN_RAM_GB,
)
from core.loaders import (
    fuse_lightning_lora_and_cast_to_fp8,  # noqa: F401  (fp8-lightning 分岐は angles 対応版手順へ置換済み。参照互換のため残置)
    fuse_lora_into_transformer,
    load_transformer_from_config,
)
from core.optimize import (
    apply_attention_backend,
    apply_compile,
    configure_transformer_offload,
    suspended_whole_module_swap_offload,
)
from core import progress as progress_mod
from core.resolve import resolve_model_path

from families.qwen_image import state
from families.qwen_image.paths import (
    EDIT_GGUF_CONFIG_REPO,
    EDIT_GGUF_FILENAME_TEMPLATE,
    EDIT_GGUF_HF_REPO,
    EDIT_LORA_ANGLES_HF_FILE,
    EDIT_LORA_ANGLES_HF_REPO,
    EDIT_LORA_ANGLES_PATH,
    EDIT_LORA_LIGHTNING_HF_FILE,
    EDIT_LORA_LIGHTNING_HF_REPO,
    EDIT_LORA_LIGHTNING_PATH,
    EDIT_PROCESSOR_REPO,
    EDIT_TRANSFORMER_FALLBACK_HF_FILE,
    EDIT_TRANSFORMER_FALLBACK_HF_REPO,
    EDIT_TRANSFORMER_FALLBACK_PATH,
    EDIT_TRANSFORMER_FALLBACK_PREFIX,
    EDIT_TRANSFORMER_HF_FILE,
    EDIT_TRANSFORMER_HF_REPO,
    EDIT_TRANSFORMER_PATH,
    BASE_REPO,
)
from families.qwen_image.runtime import is_fp8_lightning_quant, quant_suffix
from families.qwen_image.shared import (
    load_shared_components_locked,
    warn_if_model_cpu_with_quant,
)

import os

from core.resolve import COMFYUI_MODELS_DIR
from core.loaders import load_transformer_gguf


# ============================================================================
# GGUF(DS_QUANT=gguf-*)VRAM段階構成(CLAUDE.md 59番参照)
# ============================================================================
# GGUF_OFFLOAD_ALIASES: DS_OFFLOAD の明示指定値を GGUF 段階名にマップする
# (core.optimize._OFFLOAD_ALIASES と語彙を揃える。GGUF固有の3段階は
# "none"(=full常駐)/ "gguf_te_fp8"(=24GB級、text_encoderのみfp8)/ "model_cpu" の3値)。
GGUF_TIER_FULL = "none"
GGUF_TIER_TE_FP8 = "gguf_te_fp8"
GGUF_TIER_MODEL_CPU = "model_cpu"


def _resolve_gguf_offload_tier() -> str:
    """GGUF Edit transformer をどの段階でロードするかを決める。

    DS_OFFLOAD が "auto" 以外の明示値なら、それをそのまま尊重する(ユーザー指定を
    自動判定で上書きしない、core.optimize.resolve_offload_mode() と同じ流儀):
      - "none" / "off" / "cuda" 等(none系エイリアス)-> GGUF_TIER_FULL
      - "model_cpu" / "cpu" / "model" 等(model_cpu系エイリアス)-> GGUF_TIER_MODEL_CPU
      - "gguf_te_fp8"(GGUF専用の中間層、エイリアス辞書には無いため直接値のみ)
      - それ以外の明示値("group"/"group_lowvram" 等)は GGUF には未対応のため
        警告を出して自動判定にフォールバックする。

    "auto"(既定)の場合は空きVRAMで自動判定する:
      free_gb >= DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB(既定30.0)      -> full(全常駐)
      free_gb >= DS_EDIT_GGUF_MID_VRAM_FREE_THRESHOLD_GB(既定19.0)       -> te_fp8(24GB級)
      それ未満                                                            -> model_cpu(20GB級)
    """
    from core.optimize import _OFFLOAD_ALIASES  # noqa: PLC0415 (循環import回避のため遅延import)

    requested = state.get_runtime_config().offload.strip().lower().replace("-", "_")
    if requested and requested != "auto":
        normalized = _OFFLOAD_ALIASES.get(requested, requested)
        if normalized == "none":
            print(f"[families.qwen_image] GGUF offload tier: DS_OFFLOAD={requested!r} -> full(明示指定)")
            return GGUF_TIER_FULL
        if normalized == "model_cpu":
            print(f"[families.qwen_image] GGUF offload tier: DS_OFFLOAD={requested!r} -> model_cpu(明示指定)")
            return GGUF_TIER_MODEL_CPU
        if normalized == GGUF_TIER_TE_FP8:
            print(f"[families.qwen_image] GGUF offload tier: DS_OFFLOAD={requested!r} -> gguf_te_fp8(明示指定)")
            return GGUF_TIER_TE_FP8
        print(
            f"[families.qwen_image] warning: DS_OFFLOAD={requested!r} is not supported for GGUF Edit "
            f"(有効値: none/model_cpu/gguf_te_fp8); falling back to auto"
        )

    free_gb = gpu.free_vram_gb()
    if free_gb >= DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB:
        tier = GGUF_TIER_FULL
    elif free_gb >= DS_EDIT_GGUF_MID_VRAM_FREE_THRESHOLD_GB:
        tier = GGUF_TIER_TE_FP8
    else:
        tier = GGUF_TIER_MODEL_CPU
    print(
        f"[families.qwen_image] GGUF offload tier: free VRAM={free_gb:.1f}GB -> {tier} "
        f"(thresholds: full>={DS_EDIT_GGUF_FULL_VRAM_FREE_THRESHOLD_GB:.1f}GB, "
        f"te_fp8>={DS_EDIT_GGUF_MID_VRAM_FREE_THRESHOLD_GB:.1f}GB)"
    )
    return tier


def _resolve_gguf_mid_te_quant() -> str:
    """24GB級(gguf_te_fp8ティア)で共有text_encoderをどう量子化するかを決める。

    DS_EDIT_GGUF_MID_TE_QUANT(既定 "nf4")を読む。"fp8"(26GB以上空きがある環境向けの
    逃げ道、CLAUDE.md 59番参照)も選択可。未知の値は警告して既定(nf4)にフォールバックする。
    """
    value = (DS_EDIT_GGUF_MID_TE_QUANT or "nf4").strip().lower()
    if value not in ("nf4", "fp8"):
        print(
            f"[families.qwen_image] warning: DS_EDIT_GGUF_MID_TE_QUANT={value!r} は未対応です"
            f"(対応値: nf4 / fp8)。既定の nf4 にフォールバックします"
        )
        return "nf4"
    return value


def _load_edit_group_locked() -> None:
    """Edit transformer をロードし、QwenImageEditPlusPipeline を構築する(共有コンポーネントを再利用)。

    ロード順序に注意: 大きい transformer(bf16で約40GB)は「まず自分だけでVRAMに載せる」
    ことを優先し、共有コンポーネント(text_encoder等)のロードはその後に回す(先に共有
    コンポーネントをロードしてしまうと、その分VRAMの空きが減り、40GB近い transformer の
    直接ロードが収まらなくなるリスクが高まるため)。fp8-lightning は特にこの順序が重要
    (fuse時点では圧縮前のbf16、約38GBを一時的に必要とする)。

    抽出元: pipeline_manager.py _load_edit_group_locked()(行1200-1321)。
    """
    edit_group = state.edit_group
    if edit_group["loaded"]:
        return
    raw_quant = state.get_runtime_config().quant
    quant = quant_suffix(raw_quant)
    fp8_lightning = is_fp8_lightning_quant(raw_quant)
    small_transformer = bool(quant) or fp8_lightning
    t0 = time.time()
    # 注意: これはプロセス全体で一度だけ決まる「グローバル」オフロードモード
    # (state.get_offload_mode()、T2I等 他グループも参照する)。GGUF Edit専用の
    # DS_OFFLOAD=gguf_te_fp8(CLAUDE.md 59番)はこのグローバル判定には無い値のため、
    # ここで呼ぶと core.optimize.resolve_offload_mode() が
    # "unknown offload mode 'gguf_te_fp8'; falling back to auto" という警告ログを出す
    # (実機確認済み、無害だが紛らわしい)。GGUF量子化時(quant真)はこの offload_mode/
    # load_device を実際には使わない(_resolve_gguf_offload_tier() が別途正しく
    # gguf_te_fp8 を解決する)ため実害はないが、グローバル判定自体をGGUF固有の値で
    # 汚染すべきではないため、この呼び出し自体は変更せず、警告は「無視してよい既知の
    # ログ」として記録するに留める(グローバル値をここでスキップすると他グループの
    # オフロード判定タイミングが変わってしまうため、影響範囲を広げないことを優先した)。
    offload_mode = state.get_offload_mode(small_transformer_active=small_transformer)
    load_device = state.t2i_load_device(offload_mode)

    fallback_used = False
    gguf_tier = None  # GGUF_TIER_FULL / GGUF_TIER_TE_FP8 / GGUF_TIER_MODEL_CPU(quant時のみ使う)
    mid_te_quant = None  # gguf_tier==GGUF_TIER_TE_FP8 の場合のみ "nf4" / "fp8"
    nf4_te_preloaded = False
    if quant:
        gguf_tier = _resolve_gguf_offload_tier()

        # 24GB級(te_fp8ティア)+ nf4選択時のみ、共有text_encoderをGGUF transformerより
        # **先に**ロードする(2026-08-16、CLAUDE.md 59番のnf4化タスク)。
        # 理由: bnb nf4ロードは一時ピーク13.83GiBが発生する(bf16でmaterializeしてから
        # 量子化するため)。transformer(GGUF Q4_K_M 12.32GiB)を先に確保していると、
        # この一時ピークが上乗せされて24GB級カードでは高確率でOOMする。TEを先にロード
        # すれば、TE量子化完了後の常駐(5.54GiB)+一時ピークの合計でもtransformer未確保の
        # 状態のため十分な余裕がある。fp8選択時・full/model_cpuティアは従来どおり
        # transformer先行のまま(fp8のpost-hoc castingはCPU上で行うためGPU一時ピークが
        # 無く、この罠が存在しない)。
        if gguf_tier == GGUF_TIER_TE_FP8:
            mid_te_quant = _resolve_gguf_mid_te_quant()
            if mid_te_quant == "nf4" and not state.shared["loaded"]:
                runtime_config_early = state.get_runtime_config()
                original_te_quant_early = runtime_config_early.te_quant
                if original_te_quant_early in ("", "none", "off", "bf16"):
                    print(
                        "[families.qwen_image] GGUF te_fp8 tier (nf4): 共有text_encoderを"
                        "transformerより先にnf4ロードします(ロード順序の罠、CLAUDE.md 59番参照)"
                    )
                    runtime_config_early.te_quant = "nf4"
                    try:
                        load_shared_components_locked(
                            small_transformer_active=small_transformer, force_offload_mode="group"
                        )
                        nf4_te_preloaded = True
                    finally:
                        # 共有はプロセス単位のシングルトンとして既にロード済みになったため、
                        # ここで te_quant を元に戻しても shared["text_encoder"] は
                        # nf4量子化済みインスタンスのまま(後続の _apply_te_quant() 等は
                        # shared["loaded"]==True のため呼ばれない)。他ファミリー・他リクエストの
                        # 判定を汚染しないよう、fp8強制と同じ流儀で必ず元に戻す。
                        runtime_config_early.te_quant = original_te_quant_early

        filename = EDIT_GGUF_FILENAME_TEMPLATE.format(suffix=quant)
        local_path = os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", filename)
        path = resolve_model_path(local_path, EDIT_GGUF_HF_REPO, filename)
        print(f"[families.qwen_image] loading Edit transformer as GGUF({quant}) from {path} (tier={gguf_tier})")
        transformer = load_transformer_gguf(QwenImageTransformer2DModel, path, EDIT_GGUF_CONFIG_REPO)

        # GGUF量子化transformer(GGUFLinear)へのLightning LoRA適用(2026-08-16、CLAUDE.md
        # 59番: 旧CLAUDE.md 11番「GGUFにLoRA適用不可」は pipe.load_lora_weights() 経路
        # 限定の話で、transformer-level の load_lora_adapter() は現行 diffusers
        # (0.40.0.dev0)で成功することを実機確認済み。**必ずGPUへ載せる/オフロード登録する
        # 前(CPU上)で適用すること**(fp8-lightning系と同じ順序の罠、CLAUDE.md 17番)。
        # 成功時は edit_group["lora_available"]=True となり、app.py 側の steps/cfg
        # フォールバック判定(4steps/cfg1.0 側)が正しく効くようになる。
        _apply_gguf_edit_lora(transformer)
        # charsheet 用の2本目のアダプタ(fal 2511 Multiple-Angles LoRA)を同じ手順で
        # 追加ロードする(本サーバ独自の設計、モジュール冒頭のdocstring参照。
        # 必ず lightning と同様 .to("cuda") より前、CPU上で適用する)。
        _apply_angles_adapter(transformer)

        if gguf_tier == GGUF_TIER_MODEL_CPU:
            # 20GB級: transformer は enable_model_cpu_offload() に任せるため、ここでは
            # CPU 常駐のままにする(from_single_file() の既定どおり、.to("cuda") しない)。
            # RAM ガード(CLAUDE.md 33番の趣旨に当たらない根拠は _load_edit_group_locked()
            # 冒頭のモジュール docstring 追記、および CLAUDE.md 59番参照): GGUF(実測
            # 12.32GiB常駐)を生成ごとに1往復させるだけで、33番が禁止する bf16 38GB丸ごと
            # スワップの1/3規模。ホストRAM 94GB(空き85GB、59番検証時点)は33番の事故当時
            # (RAM 62GB)から増設済み。念のため同じ流儀でロード前にガードする。
            available_ram = gpu.available_ram_gb()
            if available_ram is not None and available_ram < DS_EDIT_GGUF_MODEL_CPU_MIN_RAM_GB:
                raise RuntimeError(
                    f"GGUF Edit(enable_model_cpu_offload()方式)のロードには約"
                    f"{DS_EDIT_GGUF_MODEL_CPU_MIN_RAM_GB:.0f}GB以上のホストRAM空きを推奨しますが、"
                    f"現在の空きRAM(MemAvailable)は {available_ram:.1f}GB しかありません。"
                    "他のプロセスを終了するか、DS_EDIT_GGUF_MODEL_CPU_MIN_RAM_GB を調整してください。"
                )
            transformer_offload_mode = "model_cpu"
        else:
            # full / te_fp8 のいずれも transformer 自体は常にフルGPU常駐にする
            # (GGUFは小さいためCPU RAM常駐を避ける。group系オフロードは使わない)。
            transformer.to("cuda")
            transformer_offload_mode = "none"
    elif fp8_lightning:
        # bf16 2511 transformerを直接GPUへロード(実測 約38GB)し、その場でLightning
        # LoRAをfuseしてから enable_layerwise_casting でストレージをfp8化する(実測19.05GB)。
        # 共有コンポーネント読み込み前にVRAMの空きを最大限確保するため、offload_modeに
        # 関わらず常に直接 "cuda:0" へロードする(group系オフロードは使わない)。
        path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
        print(f"[families.qwen_image] loading Edit transformer for fp8-lightning fuse from {path} (直接cuda:0へ)")
        transformer = load_transformer_from_config(
            QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, "cuda:0"
        )
        # 2026-08-18変更(diffusers-image-server 新charsheet方式): 旧実装の
        # fuse_lightning_lora_and_cast_to_fp8()(fuse + fp8キャストを一括)を、
        # 「Lightning fuse -> Multiple-angles を adapter としてロード -> fp8 layerwise cast」
        # の3段階に分解した。angles adapter は **必ず enable_layerwise_casting() より先に**
        # ロードすること(逆順だと PEFT が新規作成する lora_A/lora_B がfp8で作られ
        # forward 時に addmm_cuda NotImplementedError で落ちる。元リポジトリ
        # core/loaders.py load_lora_adapters_and_cast_to_fp8() の実機検証済みの罠)。
        print("[families.qwen_image] fusing Lightning LoRA (then angles adapter + fp8 cast)...")
        fuse_lora_into_transformer(
            transformer,
            EDIT_LORA_LIGHTNING_PATH,
            EDIT_LORA_LIGHTNING_HF_REPO,
            EDIT_LORA_LIGHTNING_HF_FILE,
            adapter_name="lightning_fuse",
            weight=1.0,
        )
        _apply_angles_adapter(transformer)
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"[families.qwen_image] fp8-lightning transformer ready, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB")
        transformer_offload_mode = "none"
    else:
        try:
            path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
            print(f"[families.qwen_image] loading Edit transformer from {path} (load_device={load_device})")
            transformer = load_transformer_from_config(
                QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, load_device
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[families.qwen_image] primary Edit transformer failed ({exc!r}); falling back to 2509 fp8")
            fallback_used = True
            path = resolve_model_path(
                EDIT_TRANSFORMER_FALLBACK_PATH, EDIT_TRANSFORMER_FALLBACK_HF_REPO, EDIT_TRANSFORMER_FALLBACK_HF_FILE
            )
            transformer = load_transformer_from_config(
                QwenImageTransformer2DModel,
                EDIT_PROCESSOR_REPO,
                "transformer",
                path,
                load_device,
                strip_prefix=EDIT_TRANSFORMER_FALLBACK_PREFIX,
            )
        transformer_offload_mode = offload_mode
        # Multiple-angles LoRA(新charsheet方式)。オフロード登録(configure_transformer_offload)
        # より前・CPU/ロードデバイス上で適用する(GGUF/fp8-lightning 分岐と同じ順序の罠)。
        _apply_angles_adapter(transformer)

    # 24GB級(gguf_te_fp8): 共有 text_encoder を量子化して圧縮する(既定nf4・実測5.54GiB、
    # DS_EDIT_GGUF_MID_TE_QUANT=fp8明示時は実測8.74GiB)。text_encoder は t2i/edit 系
    # パイプライン間で共有されるプロセス単位のシングルトン(CLAUDE.md 8番)で、
    # DS_QWEN_TE_QUANT が既に明示指定されていれば(shared.py の _apply_te_quant() /
    # load_shared_components_locked() が RuntimeConfig.te_quant をそのまま読むため)
    # そちらを尊重する。共有コンポーネントが既にロード済みの場合、この場で切り替えても
    # 他の呼び出し元(T2I等)は既に別dtypeのtext_encoderインスタンスを参照しているため、
    # 安全に「未ロード時のみ」te_quantを一時的に切り替える(呼び出し側の
    # RuntimeConfigインスタンスは1プロセス1個のシングルトンのため、他ファミリー呼び出しにも
    # 波及する点に注意。GGUF Edit(20GB/24GB級)運用時は他ファミリーもVRAMに乗らない
    # 前提のため実害は無いと判断)。
    # 注意: nf4 選択時は上の「早期プリロード」ブロックで既に shared["loaded"]=True に
    # なっているため、以下の分岐は自然に素通りする(nf4_te_preloaded の状態には依存しない、
    # state.shared["loaded"] を見れば十分)。
    runtime_config = state.get_runtime_config()
    original_te_quant = runtime_config.te_quant
    forced_te_quant = None
    if (
        gguf_tier == GGUF_TIER_TE_FP8
        and mid_te_quant == "fp8"
        and not state.shared["loaded"]
        and original_te_quant in ("", "none", "off", "bf16")
    ):
        print(
            "[families.qwen_image] GGUF te_fp8 tier: 共有text_encoderをfp8化します"
            "(DS_EDIT_GGUF_MID_TE_QUANT=fp8明示指定のため一時的にfp8へ切替、24GB級カード対応)"
        )
        runtime_config.te_quant = "fp8"
        forced_te_quant = "fp8"

    # 20GB級(model_cpu): 共有コンポーネントの配置もこのロード専用に "model_cpu" を強制する
    # (プロセス単位のグローバル offload_mode が偶々 "group"/"group_lowvram" 等に解決されて
    # いた場合、configure_shared_offload() が text_encoder に accelerate hook を付けてしまい、
    # 後続の pipe.enable_model_cpu_offload() と衝突するのを防ぐ。shared.py の
    # load_shared_components_locked() force_offload_mode 引数のdocstring参照)。
    # 24GB級(te_fp8)も同じ理由で "group" を強制する: transformer は常にGPU常駐にする
    # 設計(GGUF_TIER_TE_FP8 分岐、上記コメント参照)なので、グローバルモードが境界値付近で
    # "group_lowvram"(text_encoderをスワップ用hookで管理)に解決された場合でも、
    # 「transformer常駐+text_encoderもGPU常駐(fp8化済みなので軽い)」という
    # この層の設計を優先する("group" は vae.to("cuda"); text_encoder.to("cuda") のみで
    # accelerate hookを一切使わないため、fp8化後の素朴なGPU常駐と衝突しない)。
    if gguf_tier == GGUF_TIER_MODEL_CPU:
        shared_force_offload_mode = "model_cpu"
    elif gguf_tier == GGUF_TIER_TE_FP8:
        shared_force_offload_mode = "group"
    else:
        shared_force_offload_mode = None

    # transformer をVRAMに確保した後で共有コンポーネント(text_encoder等)をロードする
    # (上記の順序に関する注意を参照)。
    try:
        load_shared_components_locked(
            small_transformer_active=small_transformer, force_offload_mode=shared_force_offload_mode
        )
    finally:
        if forced_te_quant is not None:
            runtime_config.te_quant = original_te_quant
    warn_if_model_cpu_with_quant(quant or ("fp8-lightning" if fp8_lightning else None), offload_mode, gguf_tier=gguf_tier)

    proc_dir = snapshot_download(repo_id=EDIT_PROCESSOR_REPO, allow_patterns=["processor/*"])
    # 注意: AutoProcessor.from_pretrained() はこの環境(transformers 5.x系)だと
    # preprocessor_config.json 内の "processor_class": "Qwen2VLProcessor" を正しく解決できず、
    # 画像処理を持たない素の Qwen2Tokenizer にフォールバックしてしまう(生成時に
    # "pixel_values" KeyError で落ちる。実機で確認済み)。Qwen2VLProcessor を明示指定する。
    processor = Qwen2VLProcessor.from_pretrained(proc_dir, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")

    if transformer_offload_mode == "model_cpu":
        edit_pipe = QwenImageEditPlusPipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            processor=processor,
            transformer=transformer,
        )
        edit_pipe.enable_model_cpu_offload()
    else:
        if not quant and not fp8_lightning:  # quant/fp8_lightning時は上ですでに .to("cuda") 済み
            configure_transformer_offload(transformer, transformer_offload_mode)
        edit_pipe = QwenImageEditPlusPipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            processor=processor,
            transformer=transformer,
        )

    apply_attention_backend(transformer, state.get_runtime_config().attention_backend, state.get_runtime_config().device)
    apply_compile(transformer, state.get_runtime_config().compile)

    quant_label = "fp8-lightning" if fp8_lightning else quant
    if fp8_lightning:
        # Lightning LoRAはすでに重みにfuse済み。追加のLoRAロードは不要で、常にLightning
        # 適用状態として扱う(無効化不可)。
        edit_group["lora_available"] = True
        edit_group["lightning_merged"] = True
    elif quant:
        # GGUFは transformer-level で既に _apply_gguf_edit_lora() 済み(上記参照)。
        # そちらが edit_group["lora_available"] と初期無効化(transformer.disable_lora())を
        # 済ませているため、ここでは pipeline-level API(set_edit_lightning() が使う
        # pipe.set_adapters()/pipe.disable_lora())と状態を揃えるための保険のみ行う。
        if edit_group.get("lora_available"):
            try:
                edit_pipe.disable_lora()
            except Exception as e:  # noqa: BLE001
                print(f"[families.qwen_image] warning: GGUF Edit LoRA の初期無効化に失敗しました: {e}")
    else:
        _apply_edit_loras(edit_pipe, gguf_quantized=False)

    edit_group["transformer"] = transformer
    edit_group["edit_pipe"] = edit_pipe
    edit_group["scheduler"] = scheduler
    edit_group["processor"] = processor
    edit_group["loaded"] = True
    edit_group["load_time_s"] = time.time() - t0
    edit_group["fallback"] = fallback_used
    edit_group["quant"] = quant_label
    edit_group["gguf_tier"] = gguf_tier  # None(非GGUF) / "none" / "gguf_te_fp8" / "model_cpu"
    edit_group["gguf_mid_te_quant"] = mid_te_quant  # gguf_tier=="gguf_te_fp8" の場合のみ "nf4"/"fp8"
    print(
        f"[families.qwen_image] Edit group loaded in {edit_group['load_time_s']:.1f}s "
        f"(offload_mode={transformer_offload_mode}, fallback={fallback_used}, quant={quant_label}, "
        f"gguf_tier={gguf_tier}, gguf_mid_te_quant={mid_te_quant})"
    )


def _apply_edit_loras(pipe, gguf_quantized: bool = False) -> None:
    """Edit 用 Lightning 4steps LoRA をロードし、既定は無効化しておく(非GGUF経路専用)。

    2026-08-16修正(CLAUDE.md 59番・旧11番の訂正): GGUF量子化 transformer への
    LoRA適用は `pipe.load_lora_weights()` 経由(この関数)では失敗するが、これは
    「GGUFにLoRAを一切適用できない」ことの証明ではなかった。transformer-level の
    `load_lora_adapter()` を使えば成功することを実機確認済み(_apply_gguf_edit_lora()
    参照)。そのため GGUF 経路はこの関数を通らなくなった(呼び出し側
    _load_edit_group_locked() 参照)。gguf_quantized 引数は後方互換のため残すが、
    現在の呼び出し元は常に False で呼ぶ。

    抽出元: pipeline_manager.py _apply_edit_loras()(行1324-1349)。
    """
    edit_group = state.edit_group
    try:
        path = resolve_model_path(EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE)
        # group_lowvram の text_encoder 手動スワップを一時解除してからロードする
        # (t2i.py の _apply_t2i_loras() と同じ理由。詳細は
        # core.optimize.suspended_whole_module_swap_offload() のdocstring参照)。
        with suspended_whole_module_swap_offload(pipe.text_encoder):
            pipe.load_lora_weights(path, adapter_name="lightning")
        pipe.disable_lora()  # 既定は無効
        edit_group["lora_available"] = True
    except Exception as e:  # noqa: BLE001
        edit_group["lora_available"] = False
        edit_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        if gguf_quantized:
            print(
                f"[families.qwen_image] warning: GGUF量子化transformerへのEdit Lightning LoRA適用に失敗しました"
                f"(pipe.load_lora_weights()経路。通常は _apply_gguf_edit_lora() の"
                f"transformer-level経路が使われるはずのため、この分岐に来ること自体が想定外)。"
                f"LoRAなしにフォールバックします: {type(e).__name__}: {e}"
            )
        else:
            print(f"[families.qwen_image] warning: Edit Lightning LoRA のロードに失敗しました: {e}")


def _apply_gguf_edit_lora(transformer) -> None:
    """GGUF量子化 Edit transformer(GGUFLinear層)へ Lightning 4steps LoRA を適用する。

    2026-08-16追加(CLAUDE.md 59番。旧CLAUDE.md 11番「GGUF量子化transformerにはLoRAを
    適用できない」の訂正): 失敗するのは `pipe.load_lora_weights()` 経路のみで、
    transformer-level の `load_lora_adapter()` は現行 diffusers(0.40.0.dev0)で
    GGUFLinear に対しても成功することを実機確認済み(スタンドアロン probe で実証、
    scratchpad/probe_gguf_edit.py 参照)。

    **必ず CPU 上(GPU へ載せる/オフロード登録する前)で適用すること。**
    fp8-lightning系(core.loaders.load_lora_adapters_and_cast_to_fp8 の docstring)や
    CLAUDE.md 17番と同種の順序の罠がある可能性を考慮し、安全側に倒して
    _load_edit_group_locked() 側で `.to("cuda")` / `enable_model_cpu_offload()` の
    **前**にこの関数を呼ぶ設計にしている(実機のprobeでも同じ順序で成功を確認済み)。

    失敗時はエラーにせず警告ログを出し、lora_available=False としてフォールバックする
    (呼び出し側 set_edit_lightning() がこれを見て無効化する。app.py 側で steps/cfg を
    通常品質側にフォールバックさせる、旧 _apply_edit_loras() と同じ防御方針を維持)。
    """
    edit_group = state.edit_group
    try:
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers
        from safetensors.torch import load_file

        lora_path = resolve_model_path(
            EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE
        )
        converted_sd = _convert_non_diffusers_qwen_lora_to_diffusers(load_file(lora_path))
        transformer.load_lora_adapter(converted_sd, adapter_name="lightning", prefix="transformer")
        # 既定は無効化しておく(有効化/無効化の実際のトグルは pipeline 構築後、
        # set_edit_lightning() が pipe.set_adapters()/pipe.disable_lora() で行う。
        # ここでは load_lora_adapter() 直後の状態(有効)のまま transformer.disable_lora()
        # を呼び、pipeline-level API と一貫した初期状態にしておく)。
        transformer.disable_lora()
        edit_group["lora_available"] = True
        print("[families.qwen_image] GGUF Edit transformer: Lightning LoRA applied (transformer-level adapter)")
    except Exception as e:  # noqa: BLE001
        edit_group["lora_available"] = False
        edit_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        print(
            f"[families.qwen_image] warning: GGUF Edit transformer への Lightning LoRA適用に失敗しました。"
            f"LoRAなしにフォールバックします(steps/cfgを通常品質側に自動調整): {type(e).__name__}: {e}"
        )


def _apply_angles_adapter(transformer) -> None:
    """Multiple-angles LoRA(fal 2511、charsheet 用)を transformer-level adapter として
    ロードする(diffusers-image-server の新charsheet方式。probe_gguf_charsheet.py で
    GGUF Q4_K_M + 2アダプタ(lightning+angles)の品質・速度を実証済み)。

    呼び出し位置の規約(必ず守ること):
      - GGUF 経路: `.to("cuda")` / `enable_model_cpu_offload()` より前(CPU上)
      - fp8-lightning 経路: Lightning fuse 後・`enable_layerwise_casting()` より前
      - 非量子化 bf16 経路: `configure_transformer_offload()` より前
    (元リポジトリ CLAUDE.md 17番・59番のロード順序の罠と同一の理由)。

    fal 2511 LoRA は既に "transformer." プレフィックス付き PEFT 形式(1680キー)のため
    _convert_non_diffusers_qwen_lora_to_diffusers() は実質 no-op で安全に通過する
    (DS_ANGLES_LORA=2509 の旧LoRAも同関数+prefix="transformer" で動作する、
    元リポジトリ core/loaders.py fuse_lora_into_transformer() の docstring 参照)。

    失敗時はエラーにせず angles_available=False で通常 Edit にフォールバックする
    (charsheet 生成時に set_edit_angles() が False を返し、呼び出し側が
    RuntimeError にする)。
    """
    edit_group = state.edit_group
    try:
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers
        from safetensors.torch import load_file

        angles_path = resolve_model_path(
            EDIT_LORA_ANGLES_PATH, EDIT_LORA_ANGLES_HF_REPO, EDIT_LORA_ANGLES_HF_FILE
        )
        angles_sd = _convert_non_diffusers_qwen_lora_to_diffusers(load_file(angles_path))
        transformer.load_lora_adapter(angles_sd, adapter_name="angles", prefix="transformer")
        # 既定は無効(通常 Edit では angles を効かせない)。有効化は set_edit_angles() が
        # enable_lora() -> set_adapters([...]) の順で行う(CLAUDE.md 62番の no-op バグ対策)。
        transformer.disable_lora()
        edit_group["angles_available"] = True
        print("[families.qwen_image] Multiple-angles LoRA adapter loaded (disabled by default)")
    except Exception as e:  # noqa: BLE001
        edit_group["angles_available"] = False
        edit_group["angles_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        print(
            f"[families.qwen_image] warning: Multiple-angles LoRA adapter のロードに失敗しました"
            f"(charsheet は利用不可、通常Editは影響なし): {type(e).__name__}: {e}"
        )


def set_edit_angles(pipe, enabled: bool) -> bool:
    """charsheet 用 Multiple-angles adapter のトグル(新charsheet方式)。

    enabled=True(charsheet 生成時):
      - fp8-lightning(lightning_merged=True、Lightning は重みに fuse 済み):
        angles adapter のみ有効化(set_adapters(["angles"], [1.0]))
      - GGUF / bf16 adapter 構成: lightning + angles の2アダプタ同時有効化
        (probe と同じ weights=[1.0, 1.0])
    enabled=False(通常 Edit へ戻す):
      - fp8-lightning: disable_lora()(angles を計算から外す。Lightning は fuse 済みのため影響なし)
      - GGUF: 何もしない(直前の set_edit_lightning() が set_adapters(["lightning"]) /
        disable_lora() で active set を確定済みのため。set_adapters はアクティブ集合を
        置き換えるので angles は自動的に外れている)

    重要(CLAUDE.md 62番・63番): 有効化は必ず enable_lora() -> set_adapters() の順。
    set_adapters() は PEFT の _disable_adapters フラグをクリアしないため、
    enable_lora() を先に呼ばないと LoRA が forward に一切効かない完全な no-op になる。
    """
    edit_group = state.edit_group
    if not edit_group.get("angles_available"):
        return False
    try:
        if enabled:
            pipe.enable_lora()
            if edit_group.get("lightning_merged"):
                pipe.set_adapters(["angles"], adapter_weights=[1.0])
            else:
                pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
            return True
        if edit_group.get("lightning_merged"):
            pipe.disable_lora()
        # GGUF/bf16 adapter 構成の無効化は set_edit_lightning() に委ねる(上記docstring)。
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[families.qwen_image] warning: Multiple-angles adapter 切替に失敗しました: {e}")
        return False


def set_edit_lightning(pipe, enabled: bool) -> bool:
    """LoRA未ロード/非対応(GGUF量子化等)の場合は常に False を返す(要求されても無効のまま)。
    fp8-lightning(重みにfuse済み)の場合は常に True を返す(無効化不可)。

    抽出元: pipeline_manager.py set_edit_lightning()(行1352-1368)。
    """
    edit_group = state.edit_group
    if edit_group.get("lightning_merged"):
        return True
    if not edit_group.get("lora_available"):
        return False
    try:
        if enabled:
            # enable_lora() 必須(CLAUDE.md 62番のno-opバグ修正): ロード時の disable_lora()
            # が立てる PEFT の _disable_adapters フラグを set_adapters() はクリアしない
            # (PEFT BaseTunerLayer.set_adapter() はアクティブアダプタを変えるだけで、
            # forward() 冒頭の disable_adapters 判定はそのまま)。enable_lora() を先に
            # 呼ばないと LoRA が forward に一切効かない完全な no-op になる(実機で
            # lightning=true/false のピクセル完全一致を確認済み)。
            pipe.enable_lora()
            pipe.set_adapters(["lightning"], adapter_weights=[1.0])
        else:
            pipe.disable_lora()
        return enabled
    except Exception as e:  # noqa: BLE001
        print(f"[families.qwen_image] warning: Edit Lightning LoRA 切替に失敗しました: {e}")
        return False


def set_scheduler_shift(scheduler, shift) -> None:
    """scheduler の shift(dynamic shifting用)を上書きする。None なら何もしない。

    抽出元: pipeline_manager.py set_scheduler_shift()(行1371-1375)。

    2026-07-18バグ修正(T2I fp8-lightning 4steps の霧がかかったような品質崩壊の根本原因):
    Qwen/Qwen-Image のscheduler_config.jsonは use_dynamic_shifting=true。diffusersの
    FlowMatchEulerDiscreteScheduler.set_timesteps() は use_dynamic_shifting=True の場合、
    `self.shift`(=scheduler.config["shift"])を一切参照せず、代わりに呼び出し側
    パイプライン(pipeline_qwenimage.py等)が解像度依存で計算した mu を使って
    `sigmas = exp(mu) / (exp(mu) + (1/t - 1))` で時間シフトする(scheduling_flow_match_
    euler_discrete.py の set_timesteps/_time_shift_exponential 参照)。この mu は
    scheduler.config の base_shift/max_shift/base_image_seq_len/max_image_seq_len から
    calculate_shift() で算出され、"shift" キーは完全に無視される。そのため旧実装
    (scheduler.config["shift"] = float(shift) のみ)は fp8-lightning T2I 時に
    LIGHTNING_SHIFT=3.0 を指定していたつもりが実際には何の効果もなく、実際の有効shift
    (exp(mu))は解像度依存の値(1024²で約2.0、624²で約1.76)になっていた
    ---Lightning LoRA(4steps distilled)は特定のshiftで学習されているため、この
    ズレが「霧がかかったような品質崩壊」の直接原因だった(実機再現・特定済み)。
    Edit(2511)は参照画像を含めてimage_seq_lenが大きくなりやすく、たまたま有効shiftが
    3.0に近い値になっていたため症状が目立たなかった(charsheet 2-3枚参照時は
    mu→exp(mu)≈3.0〜3.7で偶然近似していた)。

    lightx2v公式のQwen-Image-Lightningモデルカード(diffusers使用例)は
    base_shift=max_shift=math.log(3) と指定することで、use_dynamic_shifting=True の
    ままでも解像度に依存しない一定の有効shift(=exp(mu)=3.0)を実現している
    (base_shift==max_shiftだとcalculate_shiftの線形補間が定数になるため)。
    本修正はこれに倣い、use_dynamic_shifting が有効なら base_shift/max_shift を両方
    math.log(shift) に上書きする(mu = log(shift) で固定され、time_shift後の有効shiftが
    ちょうど shift になる)。use_dynamic_shifting が無効なスケジューラでは従来通り
    "shift" キーを直接上書きする(非dynamic分岐は self.shift を直接使うため)。
    """
    if shift is None:
        return
    shift = float(shift)
    if scheduler.config.get("use_dynamic_shifting", False):
        log_shift = math.log(shift)
        scheduler.config["base_shift"] = log_shift
        scheduler.config["max_shift"] = log_shift
    else:
        scheduler.config["shift"] = shift


def _switch_edit_quant_if_needed(quant: "str | None") -> None:
    """リクエストで明示された quant(例: "gguf-q4_k_m" / "fp8-lightning" / "none")に
    応じて、現在アクティブな量子化方式(state.get_runtime_config().quant)を切り替える
    (CLAUDE.md 69番、t2i.py の _switch_t2i_quant_if_needed() と同じパターン)。

    注意: quant は RuntimeConfig 上で t2i と edit が同じフィールドを共有しているため、
    「これからロードする側(edit)」に対して切り替える。Edit のロード自体は
    family.py 側で「fuse系ならT2Iグループを事前解放する」既存ロジック(32番・66番)を
    経由するため、他グループの事前解放は family.py に任せ、ここでは edit グループの
    unload のみ行う。

    quant が None(runtime.normalize_quant_request() で正規化済み、"default"/未指定を含む)
    の場合は何もしない(現行の DS_QUANT 既定のまま)。
    """
    if quant is None:
        return
    from families.qwen_image.runtime import gguf_quant_vram_ok

    current = state.get_runtime_config().quant
    if quant == current:
        return
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
        current = state.get_runtime_config().quant
        if quant == current:
            return
        print(f"[families.qwen_image] switching Edit quant: {current!r} -> {quant!r}")
        state.get_runtime_config().quant = quant
        need_switch = True

    if need_switch:
        lifecycle.unload("edit")


def get_edit_pipeline(quant: "str | None" = None) -> QwenImageEditPlusPipeline:
    """抽出元: pipeline_manager.py get_edit_pipeline()(行1532-1537)。

    quant(CLAUDE.md 69番、2026-08-18追加): 明示指定時、現在アクティブな量子化方式と
    異なれば自動的に切り替える(bf16/fp8-lightning ⇔ GGUF)。
    """
    _switch_edit_quant_if_needed(quant)
    if not state.edit_group["loaded"]:
        with state.lock:
            if not state.edit_group["loaded"]:
                _load_edit_group_locked()
    pipe = state.edit_group["edit_pipe"]
    progress_mod.disable_diffusers_tqdm(pipe)
    return pipe


def get_edit_i2i_pipeline(quant: "str | None" = None) -> QwenImageImg2ImgPipeline:
    """Edit(2511)transformer を参照共有した QwenImageImg2ImgPipeline を返す(CLAUDE.md 64番)。

    背景: Edit-2511 transformer を QwenImageImg2ImgPipeline に参照共有で流し込んだ
    「Edit モデルの I2I」は、無印 Qwen-Image I2I と同等の strength 応答カーブ・同等品質で
    動くことを実機 A/B で確認済み(scratchpad/probe_edit2511_i2i.py)。追加 VRAM は
    ゼロ(vae/text_encoder/tokenizer/transformer はすべて edit_pipe と同一インスタンスを
    参照するだけで、コピーは一切行わない)。

    scheduler だけは `FlowMatchEulerDiscreteScheduler.from_config(edit_pipe.scheduler.config)`
    で独立インスタンスにする(probe と同じ。QwenImageEditPlusPipeline と
    QwenImageImg2ImgPipeline が同じ scheduler インスタンスの set_timesteps() 状態を
    奪い合わないようにするため)。

    まず get_edit_pipeline() で Edit グループのロードを確定させてから呼ぶこと
    (このモジュールの `_locked` 系関数の呼び出し規約(CLAUDE.md 14番)に従い、
    ロードそのものはこの関数では行わない)。`state.edit_group["i2i_pipe"]` に
    キャッシュし、Edit グループの再ロード(unload → get_edit_pipeline())が起きるたびに
    再構築が必要になるため、`lifecycle.unload("edit")` は edit_pipe と併せて
    i2i_pipe も必ず解放すること(transformer への参照が残ると VRAM が解放されない、
    CLAUDE.md 18番と同じ罠)。
    """
    edit_pipe = get_edit_pipeline(quant)  # Editグループのロードを確定させる(冪等)
    cached = state.edit_group.get("i2i_pipe")
    if cached is not None:
        progress_mod.disable_diffusers_tqdm(cached)
        return cached
    with state.lock:
        cached = state.edit_group.get("i2i_pipe")
        if cached is not None:
            progress_mod.disable_diffusers_tqdm(cached)
            return cached
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(edit_pipe.scheduler.config)
        i2i_pipe = QwenImageImg2ImgPipeline(
            scheduler=scheduler,
            vae=edit_pipe.vae,
            text_encoder=edit_pipe.text_encoder,
            tokenizer=edit_pipe.tokenizer,
            transformer=edit_pipe.transformer,
        )
        state.edit_group["i2i_pipe"] = i2i_pipe
    progress_mod.disable_diffusers_tqdm(i2i_pipe)
    return i2i_pipe
