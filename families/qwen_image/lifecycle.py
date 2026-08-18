# -*- coding: utf-8 -*-
"""
unload() / get_status() の共通実装(diffusers-image-server 版)。

diffusers-server からの移植。layered / edit_angles 系グループの分岐を削除した
(本リポジトリには存在しないグループのため)。解放順序(ControlNet -> T2I)は
元実装のまま維持する(ControlNetパイプラインは t2i_group["transformer"] を直接参照
しているため、順序が逆だと transformer への参照が残り VRAM が解放されない)。
"""
import gc

from core import gpu

from families.qwen_image import state


def unload(target: str = "all") -> dict:
    """VRAM 解放。target: "t2i" / "edit" / "controlnet" / "all"。

    target が "all" 以外でも、解放後に他のどのグループもロードされていなければ
    共有コンポーネント(vae/text_encoder/tokenizer)も併せて解放する
    (元リポジトリ CLAUDE.md 56番の一般化された判定)。
    """
    freed = []
    with state.lock:
        controlnet_union_group = state.controlnet_union_group
        controlnet_inpaint_group = state.controlnet_inpaint_group
        t2i_group = state.t2i_group
        edit_group = state.edit_group
        shared = state.shared

        if target in ("t2i", "controlnet", "all") and controlnet_union_group["loaded"]:
            controlnet_union_group["controlnet"] = None
            controlnet_union_group["pipe"] = None
            controlnet_union_group["scheduler"] = None
            controlnet_union_group["loaded"] = False
            freed.append("controlnet-union")
        if target in ("t2i", "controlnet", "all") and controlnet_inpaint_group["loaded"]:
            controlnet_inpaint_group["controlnet"] = None
            controlnet_inpaint_group["pipe"] = None
            controlnet_inpaint_group["scheduler"] = None
            controlnet_inpaint_group["loaded"] = False
            freed.append("controlnet-inpaint")
        if target in ("t2i", "all") and t2i_group["loaded"]:
            t2i_group["transformer"] = None
            t2i_group["t2i_pipe"] = None
            t2i_group["i2i_pipe"] = None
            t2i_group["scheduler"] = None
            t2i_group["loaded"] = False
            t2i_group["quant"] = None
            t2i_group["lora_available"] = None
            t2i_group["lora_unavailable_reason"] = None
            t2i_group["lightning_merged"] = False
            freed.append("t2i")
        if target in ("edit", "all") and edit_group["loaded"]:
            edit_group["transformer"] = None
            edit_group["edit_pipe"] = None
            # i2i_pipe は edit_pipe と transformer 等を参照共有しているため、
            # 一緒に None にしないと VRAM が解放されない(CLAUDE.md 18番/64番の罠)。
            edit_group["i2i_pipe"] = None
            edit_group["scheduler"] = None
            edit_group["processor"] = None
            edit_group["loaded"] = False
            edit_group["quant"] = None
            edit_group["lora_available"] = None
            edit_group["lora_unavailable_reason"] = None
            edit_group["lightning_merged"] = False
            edit_group["text_only_call"] = None
            edit_group["angles_available"] = None
            edit_group["angles_unavailable_reason"] = None
            freed.append("edit")
        if (
            shared["loaded"]
            and not t2i_group["loaded"]
            and not edit_group["loaded"]
        ):
            shared["vae"] = None
            shared["text_encoder"] = None
            shared["tokenizer"] = None
            shared["loaded"] = False
            freed.append("shared")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[families.qwen_image] unloaded: {freed}")
    return {"freed": freed}


def unload_shared_if_unused() -> bool:
    """他のどのグループもロードしていなければ共有コンポーネント(vae/text_encoder/
    tokenizer、bf16で~15.7GB)を解放する。解放したら True を返す。

    T2I ⇔ Edit 切替時の fuse 一時ピーク OOM 対策(元リポジトリ CLAUDE.md 22番/32番)。
    """
    freed = False
    with state.lock:
        shared = state.shared
        if (
            shared["loaded"]
            and not state.t2i_group["loaded"]
            and not state.edit_group["loaded"]
        ):
            shared["vae"] = None
            shared["text_encoder"] = None
            shared["tokenizer"] = None
            shared["loaded"] = False
            freed = True
        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    if freed:
        print("[families.qwen_image] unloaded: ['shared'] (グループ切替のため一旦解放)")
    return freed


def get_status() -> dict:
    from families.qwen_image.paths import is_2512_model

    runtime_config = state.get_runtime_config()
    t2i_group = state.t2i_group
    edit_group = state.edit_group
    controlnet_union_group = state.controlnet_union_group
    controlnet_inpaint_group = state.controlnet_inpaint_group
    shared = state.shared

    status = {
        "offload_mode": state._offload_mode,
        "runtime_config": repr(runtime_config),
        "shared_loaded": shared["loaded"],
        "shared_load_time_s": shared["load_time_s"],
        "t2i_model": "2512" if is_2512_model(runtime_config.t2i_model) else "qwen-image",
        "t2i_loaded": t2i_group["loaded"],
        "t2i_load_time_s": t2i_group["load_time_s"],
        "t2i_quant": t2i_group["quant"],
        "t2i_lora_available": t2i_group["lora_available"],
        "t2i_lora_unavailable_reason": t2i_group["lora_unavailable_reason"],
        "t2i_lightning_merged": t2i_group["lightning_merged"],
        "t2i_gguf_tier": t2i_group.get("gguf_tier"),
        "t2i_gguf_mid_te_quant": t2i_group.get("gguf_mid_te_quant"),
        "edit_loaded": edit_group["loaded"],
        "edit_load_time_s": edit_group["load_time_s"],
        "edit_transformer_fallback": edit_group["fallback"],
        "edit_quant": edit_group["quant"],
        "edit_lora_available": edit_group["lora_available"],
        "edit_lora_unavailable_reason": edit_group["lora_unavailable_reason"],
        "edit_lightning_merged": edit_group["lightning_merged"],
        "edit_gguf_tier": edit_group.get("gguf_tier"),
        "edit_gguf_mid_te_quant": edit_group.get("gguf_mid_te_quant"),
        "edit_angles_available": edit_group.get("angles_available"),
        "edit_angles_unavailable_reason": edit_group.get("angles_unavailable_reason"),
        "shared_te_quant_effective": shared.get("te_quant_effective"),
        "controlnet_union_loaded": controlnet_union_group["loaded"],
        "controlnet_union_load_time_s": controlnet_union_group["load_time_s"],
        "controlnet_inpaint_loaded": controlnet_inpaint_group["loaded"],
        "controlnet_inpaint_load_time_s": controlnet_inpaint_group["load_time_s"],
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
    }
    return status


def is_any_loaded() -> bool:
    return (
        state.shared["loaded"]
        or state.t2i_group["loaded"]
        or state.edit_group["loaded"]
        or state.controlnet_union_group["loaded"]
        or state.controlnet_inpaint_group["loaded"]
    )
