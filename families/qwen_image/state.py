# -*- coding: utf-8 -*-
"""
Qwen-Image ファミリーのシングルトン状態(グループ辞書群 + プロセス内ロック)。

diffusers-server からの移植(diffusers-image-server 版): layered / edit_angles /
edit_angles_bf16 / edit_angles_bf16group の各グループを削除した。charsheet は
edit_group + Multiple-angles adapter(edit.py の set_edit_angles())で実現する
(新charsheet方式。旧3方式は本リポジトリには存在しない)。

注意: この lock は非再入。"_load_*_locked" 関数は呼び出し側が lock を保持している
前提の規約を維持する(二重ロックしない。元リポジトリ CLAUDE.md 14番の罠)。
"""
import threading

from core.gpu import free_vram_gb
from core.optimize import resolve_offload_mode

from families.qwen_image.runtime import RuntimeConfig

# ============================================================================
# シングルトン状態
# ============================================================================
lock = threading.Lock()

_runtime_config: "RuntimeConfig | None" = None
_offload_mode: "str | None" = None  # プロセス内で一度決定したら使い回す

shared = {
    "vae": None,
    "text_encoder": None,
    "tokenizer": None,
    "loaded": False,
    "load_time_s": None,
    "te_quant_effective": None,  # ロード時に実際に使われたTE量子化方式: None(bf16)/"fp8"/"nf4"
}

# T2I/I2I グループ(無印 Qwen/Qwen-Image または Qwen/Qwen-Image-2512 のどちらか一方)。
t2i_group = {
    "transformer": None,
    "t2i_pipe": None,
    "i2i_pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,  # None(非量子化) または "Q4_K_M" 等 / "fp8-lightning"
    "lora_available": None,
    "lora_unavailable_reason": None,
    "lightning_merged": False,
    "gguf_tier": None,
    "gguf_mid_te_quant": None,
}

edit_group = {
    "transformer": None,
    "edit_pipe": None,
    "i2i_pipe": None,  # Edit transformerを参照共有したQwenImageImg2ImgPipeline(strength付きEdit-I2I)
    "scheduler": None,
    "processor": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,
    "lora_available": None,
    "lora_unavailable_reason": None,
    "lightning_merged": False,
    "gguf_tier": None,
    "gguf_mid_te_quant": None,
    "text_only_call": None,  # 画像なし(テキストのみ)生成用パッチ済み __call__
    # 新charsheet方式(diffusers-image-server): Multiple-angles LoRA を transformer-level
    # adapter として edit グループロード時に同時ロードし、リクエスト単位でトグルする
    # (edit.py の _apply_angles_adapter() / set_edit_angles())。
    "angles_available": None,
    "angles_unavailable_reason": None,
}

# ControlNet(Inpainting)。base Qwen-Image の transformer/vae/text_encoder を共有し、
# ControlNetモジュールのみ追加ロードする(t2i_group に依存。解放順序は lifecycle.unload() 参照)。
controlnet_union_group = {
    "controlnet": None,
    "pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
}

controlnet_inpaint_group = {
    "controlnet": None,
    "pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
}


def get_runtime_config() -> RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config


def get_offload_mode(small_transformer_active: bool = False) -> str:
    """オフロードモードはプロセス内で一度だけ決定し、以後のロードで使い回す
    (共有コンポーネントへの二重フック登録を避けるため)。
    """
    global _offload_mode
    if _offload_mode is None:
        config = get_runtime_config()
        free_gb = free_vram_gb()
        _offload_mode = resolve_offload_mode(config.offload, free_gb, small_transformer_active=small_transformer_active)
        print(
            f"[families.qwen_image] free VRAM: {free_gb:.1f} GB -> offload_mode={_offload_mode}"
            f"{' (小型transformer向け閾値緩和適用)' if small_transformer_active else ''}"
        )
    return _offload_mode


def t2i_load_device(offload_mode: str):
    import torch

    if offload_mode == "none":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cpu"


def reset_offload_mode_cache() -> None:
    global _offload_mode
    _offload_mode = None
