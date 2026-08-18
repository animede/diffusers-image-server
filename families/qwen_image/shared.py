# -*- coding: utf-8 -*-
"""
共有コンポーネント(vae / text_encoder / tokenizer)のロード。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_shared_components_locked()(行774-801)
  - _warn_if_model_cpu_with_quant()(行928-941)

注意: 呼び出し側が families.qwen_image.state.lock を保持している前提("_locked" 系関数の規約)。
"""
import time

import torch
from diffusers import AutoencoderKLQwenImage
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from core.optimize import configure_shared_offload

from families.qwen_image import state
from families.qwen_image.paths import BASE_REPO


# 共有 text_encoder(Qwen2_5_VLForConditionalGeneration)の fp8 layerwise casting 用
# skip パターン。diffusers の既定
# (DEFAULT_SKIP_MODULES_PATTERN = ('pos_embed', 'patch_embed', 'norm', '^proj_in$', '^proj_out$'))
# は diffusers 側 transformer の命名規則向けで、以下は拾えない:
#   - embed_tokens(nn.Embedding。layerwise casting の対象レイヤーに含まれるため、
#     指定しないとfp8化されてしまう)
#   - lm_head(nn.Linear)
#   - rotary_emb(バッファのみのモジュールだが念のため)
# 一方 'norm' は re.search の部分一致なので RMSNorm 系(input_layernorm /
# post_attention_layernorm / q_norm / k_norm 等)は既定のまま skip される。
# 'patch_embed' も vision 側 patch_embed に部分一致する。
# LTX-2.3 の LTX2_TE_FP8_SKIP_MODULES_PATTERN(families/ltx2/convert.py、CLAUDE.md 42番)と
# 同じ考え方。
QWEN_TE_FP8_SKIP_MODULES_PATTERN = (
    "pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$",  # diffusers既定を維持
    "embed_tokens", "lm_head", "rotary_emb",  # Qwen2.5-VL 固有の追加保護
)


def _apply_te_quant(text_encoder) -> None:
    """DS_QWEN_TE_QUANT に従って共有 text_encoder を量子化する(既定 "none" は何もしない)。

    **CPU 上の text_encoder に対して呼ぶこと**(GPU へ載せる前)。layerwise casting は
    呼んだ瞬間に圧縮が実行される(CLAUDE.md 12番)ので、CPU 上で圧縮してから GPU へ
    載せれば、bf16 の約16GB が GPU 上に一度も存在せずに済む。

    注意: "nf4" はこの関数では扱わない(bnb nf4 は `from_pretrained(quantization_config=...)`
    でロード時に直接量子化する必要があり、post-hoc casting方式のfp8とは構造が異なるため)。
    nf4 のロードは `load_shared_components_locked()` 側で `_load_text_encoder_nf4()` を
    使って分岐する(2026-08-16、CLAUDE.md 59番のnf4化タスク参照)。
    """
    te_quant = state.get_runtime_config().te_quant
    if te_quant in ("", "none", "off", "bf16"):
        return
    if te_quant != "fp8":
        print(
            f"[families.qwen_image] warning: DS_QWEN_TE_QUANT={te_quant!r} は post-hoc casting経路では"
            f"未対応です(対応値: none / fp8)。bf16 のままロードします"
        )
        return

    from diffusers.hooks import apply_layerwise_casting

    n_before = sum(p.numel() * p.element_size() for p in text_encoder.parameters()) / 1024**3
    apply_layerwise_casting(
        text_encoder,
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        skip_modules_pattern=QWEN_TE_FP8_SKIP_MODULES_PATTERN,
    )
    n_after = sum(p.numel() * p.element_size() for p in text_encoder.parameters()) / 1024**3
    print(
        f"[families.qwen_image] text_encoder fp8 layerwise casting applied "
        f"(DS_QWEN_TE_QUANT=fp8, パラメータ実サイズ {n_before:.2f}GB -> {n_after:.2f}GB)"
    )


def _load_text_encoder_nf4():
    """共有 text_encoder を bnb NF4 量子化で直接 GPU へロードする(2026-08-16追加、
    CLAUDE.md 59番のnf4化タスク)。

    実測(GPU:0単独占有): 常駐 5.54GiB、ただしロード中の一時ピーク 13.83GiB(bf16で
    materializeしてから量子化するため)。**この一時ピークが構造的制約になる**: GGUF Edit
    24GB級ティア(edit.py の GGUF_TIER_TE_FP8)でこの関数を使う場合、transformer
    (GGUF Q4_K_M 実測12.32GiB)を先にGPUへ確保していると、13.83GiBの一時ピークが乗って
    24GB級カードでは高確率でOOMする。そのため呼び出し側(edit.py)は
    このtierに限り「TEを先にロード→その後transformerをロード」という逆順にすること。
    """
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_REPO, subfolder="text_encoder", quantization_config=qc, dtype=torch.bfloat16, device_map={"": 0}
    )
    if torch.cuda.is_available():
        print(
            f"[families.qwen_image] text_encoder bnb nf4 loaded "
            f"(DS_QWEN_TE_QUANT=nf4, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB)"
        )
    return text_encoder


def load_shared_components_locked(small_transformer_active: bool = False, force_offload_mode: "str | None" = None) -> None:
    """vae / text_encoder / tokenizer を1回だけロードする(呼び出し側でロック保持前提)。

    抽出元: pipeline_manager.py _load_shared_components_locked()(行774-801)。ロジック無変更。

    force_offload_mode(2026-08-16追加、CLAUDE.md 59番、GGUF Edit 20GB級対応):
    通常は state.get_offload_mode()(プロセス単位でVRAM閾値から一度だけ決まる、既存の
    グローバルなオフロードモード)をそのまま使うが、GGUF の enable_model_cpu_offload()
    経路(families/qwen_image/edit.py の GGUF_TIER_MODEL_CPU)はこのグローバルモードとは
    独立に「今回のロードだけ model_cpu 相当で扱いたい」ため、明示的に上書きできるように
    した。None(既定)なら従来どおり state.get_offload_mode() を使う(挙動変更なし)。
    force_offload_mode="model_cpu" を渡すと、グローバルモードの実際の値に関わらず
    configure_shared_offload() を呼ばない(vae/text_encoderをCPU常駐のまま残し、
    呼び出し側の pipe.enable_model_cpu_offload() に配置を一任する)。これをしないと、
    グローバルモードが偶々 "group_lowvram" 等に解決された場合、configure_shared_offload()
    が text_encoder に accelerate の cpu_offload_with_hook を付けてしまい、後から呼ばれる
    pipe.enable_model_cpu_offload()(同じく accelerate hook を使う)と衝突する
    (core.optimize.suspended_whole_module_swap_offload() のdocstringに記載されている
    既知の衝突パターンと同種)。
    """
    shared = state.shared
    if shared["loaded"]:
        return
    t0 = time.time()
    offload_mode = force_offload_mode if force_offload_mode is not None else state.get_offload_mode(
        small_transformer_active=small_transformer_active
    )
    # from_pretrained() は既定で CPU にロードする。配置は configure_shared_offload() に一任する
    # (offload_mode="model_cpu" の場合はパイプライン単位の enable_model_cpu_offload() に任せる)。

    print(f"[families.qwen_image] loading shared components (vae/text_encoder/tokenizer) from {BASE_REPO}")
    vae = AutoencoderKLQwenImage.from_pretrained(BASE_REPO, subfolder="vae", torch_dtype=torch.bfloat16)

    # text_encoder のロード方式は DS_QWEN_TE_QUANT で分岐する(2026-08-16、nf4追加。
    # CLAUDE.md 59番のnf4化タスク)。nf4 は from_pretrained(quantization_config=...) で
    # ロード時に直接量子化する必要があり、fp8(post-hoc layerwise casting、CPU上で
    # bf16ロード後にキャスト)とは構造が異なるため別分岐にする。nf4 は
    # device_map={"":0} で直接GPUへロードするため(常駐5.54GiB)、以降の
    # configure_shared_offload() による .to() 呼び出しは不要かつ bnb モデルには
    # 呼ばない方が安全(core.optimize._is_bnb_quantized_text_encoder 参照)。
    te_quant = state.get_runtime_config().te_quant
    text_encoder_is_nf4 = te_quant == "nf4"
    if text_encoder_is_nf4:
        text_encoder = _load_text_encoder_nf4()
    else:
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
    tokenizer = Qwen2Tokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer")

    # 共有VAEのtiled encode/decode常時化(DS_QWEN_TILED_VAE=1既定、CLAUDE.md 56番)。
    # LTX2(CLAUDE.md 52番)と全く同じ構図: AutoencoderKLQwenImage._encode()/_decode() は
    # use_tiling を尊重してタイル分割経路(tiled_encode/tiled_decode)へ切り替わる
    # (autoencoder_kl_qwenimage.py の _encode()/_decode() 実装参照)。既定タイルサイズ
    # (256px、ストライド192px=64pxオーバーラップでブレンド)を使えば呼び出し側の変更は
    # 不要。1280x720キャンバスのControlNet Inpaint(/api/outpaint)は encode 単体で
    # 4.45GiB要求→OOMを実機確認済み(タイル閾値256pxを大きく超えるため確実にtiled化
    # される)。全パイプライン(t2i/i2i/edit/edit_angles/controlnet/controlnet_inpaint)
    # がこの共有VAEインスタンスを参照するため、ここで一度設定するだけで全モードに効く
    # (Layeredは専用VAEのため対象外。families/qwen_image/layered.py 参照、CLAUDE.md 8番)。
    if state.get_runtime_config().tiled_vae:
        vae.enable_tiling()
        print("[families.qwen_image] shared VAE tiled encode/decode enabled (DS_QWEN_TILED_VAE=1)")

    if not text_encoder_is_nf4:
        # text_encoder の fp8 化(DS_QWEN_TE_QUANT=fp8、16GB級カード対応)。
        # **必ず GPU へ載せる前(CPU 上)に実行すること。** GPU へ載せてからキャストすると、
        # 一時的に bf16 の約16GB を GPU 上に確保することになり、16GB カードでは
        # その時点で OOM する(この機能の目的そのものを達成できない)。
        _apply_te_quant(text_encoder)

    if offload_mode == "model_cpu":
        # パイプライン単位の enable_model_cpu_offload() に任せるため、ここでは配置しない。
        pass
    elif text_encoder_is_nf4:
        # nf4 は device_map={"":0} で既にGPU常駐済み。vae のみ配置する(text_encoderへの
        # .to() 呼び出しは避ける、core.optimize._is_bnb_quantized_text_encoder 参照)。
        vae.to("cuda")
    else:
        configure_shared_offload(vae, text_encoder, offload_mode)

    shared["vae"] = vae
    shared["text_encoder"] = text_encoder
    shared["tokenizer"] = tokenizer
    shared["loaded"] = True
    shared["load_time_s"] = time.time() - t0
    shared["te_quant_effective"] = "nf4" if text_encoder_is_nf4 else (te_quant if te_quant == "fp8" else None)
    print(f"[families.qwen_image] shared components loaded in {shared['load_time_s']:.1f}s")


def warn_if_model_cpu_with_quant(quant, offload_mode, gguf_tier=None) -> None:
    """DS_QUANT(GGUF) と DS_OFFLOAD=model_cpu の組み合わせは(元々は)未サポートだった。
    共有コンポーネントが CPU に取り残されないよう明示的に GPU へ配置する。

    抽出元: pipeline_manager.py _warn_if_model_cpu_with_quant()(行928-941)。

    2026-08-16変更(CLAUDE.md 59番、GGUF Edit 20GB級対応): families/qwen_image/edit.py の
    GGUF_TIER_MODEL_CPU(20GB級、enable_model_cpu_offload()方式)は「GGUF + model_cpu」の
    組み合わせを正式にサポートする新経路のため、この警告・強制GPU配置の対象から除外する
    (gguf_tier="model_cpu" の場合は何もしない)。この経路では共有コンポーネント自体も
    enable_model_cpu_offload() 経由でCPU/GPUを行き来する設計になっているため、ここで
    強制的にGPUへ固定すると20GB級カードでOOMする。他の呼び出し元(gguf_tier省略時、
    t2i.py/layered.py 等の非対応な GGUF+model_cpu 組み合わせ)は従来どおりの挙動を維持する。
    """
    if gguf_tier == "model_cpu":
        return
    shared = state.shared
    if quant and offload_mode == "model_cpu" and shared["loaded"]:
        print(
            "[families.qwen_image] warning: DS_QUANT(GGUF)とDS_OFFLOAD=model_cpuの組み合わせは"
            "未サポートです。共有コンポーネントを強制的にGPU常駐にします。"
        )
        if shared["vae"] is not None:
            shared["vae"].to("cuda")
        if shared["text_encoder"] is not None:
            shared["text_encoder"].to("cuda")
