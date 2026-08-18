# -*- coding: utf-8 -*-
"""
生成本体(T2I / I2I / Edit / Edit-I2I / Edit-charsheet / ControlNet / Inpaint / Canny)。

diffusers-server からの移植(diffusers-image-server 版)。layered / edit_angles 系
3変種の run_* を削除し、代わりに run_edit_charsheet()(Edit グループ +
Multiple-angles adapter トグル、edit.py の set_edit_angles())を追加した。

この関数群は「画像(PIL.Image 等)は既に読み込み・前処理済み」という前提で呼ばれる。
生成の排他制御(GPU 同時1件)は呼び出し側(app.py の _generate_or_409() /
apps.charsheet.jobs)が担う。

重要(angles adapter の状態リセット): fp8-lightning 構成では Lightning が重みに
fuse 済みのため、set_edit_lightning() はアダプタ状態を一切触らない。charsheet
リクエストで有効化した angles adapter が次の通常 Edit リクエストに残留しないよう、
run_edit()/run_edit_i2i() は必ず set_edit_angles(pipe, False) を呼ぶ
(GGUF 構成では set_edit_lightning() が set_adapters(["lightning"]) で active set を
置き換えるため自動的に外れるが、fp8-lightning 構成ではこの明示リセットが唯一の経路)。
"""
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

from core import progress as progress_mod
from core.optimize import (
    disable_text_encoder_cpu_offload,
    enable_text_encoder_cpu_offload,
    ensure_text_encoder_on_gpu,
    should_offload_edit_text_encoder,
)

from families.qwen_image import controlnet as controlnet_mod
from families.qwen_image import edit as edit_mod
from families.qwen_image import edit_text_only as edit_text_only_mod
from families.qwen_image import state, t2i as t2i_mod
from families.qwen_image.paths import (
    LIGHTNING_SHIFT,
    LIGHTNING_TRUE_CFG_SCALE,
    NEGATIVE_PROMPT_DEFAULT,
    is_2512_model,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MAX_EDIT_IMAGES = 3


def _round16(x: int) -> int:
    return max(16, round(x / 16) * 16)


def _new_output_path(mode: str, ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def preprocess_image(image: Image.Image, target_pixels: int = 1024 * 1024) -> Image.Image:
    """ComfyUI の ImageScaleToTotalPixels 相当。アスペクト比維持で総画素数を揃え、16の倍数に丸める。"""
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")
    scale = (target_pixels / (w * h)) ** 0.5
    new_w = _round16(w * scale)
    new_h = _round16(h * scale)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def make_generator(seed: Optional[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None or seed < 0:
        return None, None
    return torch.Generator(device=device).manual_seed(seed), seed


def t2i_model_meta() -> dict:
    model = "2512" if is_2512_model(state.get_runtime_config().t2i_model) else "qwen-image"
    return {"model": model, "quant": state.t2i_group["quant"]}


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None


def _progress_callback(mode: str, total_steps: int, extra: "dict | None" = None):
    """diffusers の callback_on_step_end に渡すクロージャを作る(step_index は 0-origin)。"""
    progress_mod.start_generating(mode, total_steps, extra)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def run_t2i(req: dict) -> dict:
    pipe = t2i_mod.get_t2i_pipeline(req.get("model"), req.get("quant"))

    # T2I は I2I/Edit系と state.shared["text_encoder"] を共有している。直前の呼び出しが
    # TE を CPU 退避したまま戻ってきた場合の device mismatch 対策(元 CLAUDE.md 27番)。
    ensure_text_encoder_on_gpu(pipe)

    steps = req["steps"]
    cfg = req["cfg"]
    shift = req.get("shift")
    lightning_active = False
    lightning_unavailable = False
    if req.get("lightning"):
        req_steps = steps if steps in (4, 8) else 4
        lightning_active = t2i_mod.set_t2i_lightning(pipe, req_steps)
        if lightning_active:
            steps = req_steps
            cfg = LIGHTNING_TRUE_CFG_SCALE
            shift = shift if shift is not None else LIGHTNING_SHIFT
        else:
            lightning_unavailable = True
            if steps <= 8:
                steps = 30
            if cfg <= 1.5:
                cfg = 4.0
    else:
        t2i_mod.set_t2i_lightning(pipe, -1)

    edit_mod.set_scheduler_shift(pipe.scheduler, shift)

    generator, used_seed = make_generator(req.get("seed", -1))
    width = _round16(req["width"])
    height = _round16(req["height"])

    if should_offload_edit_text_encoder(width, height, pipe=pipe):
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator,
        callback_on_step_end=_progress_callback("t2i", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("t2i")
    image.save(out_path)

    meta = {
        "mode": "t2i",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "lightning_requested": req.get("lightning", False),
        "lightning_unavailable_reason": (
            "量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
    }
    meta.update(t2i_model_meta())
    return meta


def run_i2i(req: dict, src_image: Image.Image) -> dict:
    """src_image は preprocess_image() 済みを渡すこと。TE CPU退避は run_edit() と同じ判定
    (未適用だと VAE decode の一時ピークで OOM する、元 CLAUDE.md 26番/27番)。"""
    pipe = t2i_mod.get_i2i_pipeline(req.get("model"), req.get("quant"))

    steps_eff = req["steps"]
    cfg_eff = req["cfg"]
    shift_eff = req.get("shift")
    lightning_active = False
    lightning_unavailable = False
    if req.get("lightning"):
        req_steps = steps_eff if steps_eff in (4, 8) else 4
        lightning_active = t2i_mod.set_t2i_lightning(pipe, req_steps)
        if lightning_active:
            steps_eff = req_steps
            cfg_eff = LIGHTNING_TRUE_CFG_SCALE
            shift_eff = shift_eff if shift_eff is not None else LIGHTNING_SHIFT
        else:
            lightning_unavailable = True
            if steps_eff <= 8:
                steps_eff = 30
            if cfg_eff <= 1.5:
                cfg_eff = 4.0
    else:
        t2i_mod.set_t2i_lightning(pipe, -1)

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    src_w, src_h = src_image.size
    te_offload = should_offload_edit_text_encoder(src_w, src_h, pipe=pipe)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        # 共有 text_encoder が別パイプライン経由で CPU 退避されたままのケースの保険。
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        image=src_image,
        strength=req.get("strength", 0.6),
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        generator=generator,
        callback_on_step_end=_progress_callback("i2i", steps_eff),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("i2i")
    out_image.save(out_path)

    meta = {
        "mode": "i2i",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "strength": req.get("strength", 0.6),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "input_size": list(src_image.size),
        "te_offload": te_offload,
        "quant": state.t2i_group["quant"],
        "lightning_requested": req.get("lightning", False),
        "lightning_unavailable_reason": (
            "GGUF量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
    }
    return meta


def run_edit(req: dict, src_images: list) -> dict:
    """通常 Edit(指示編集 / テキストのみ生成 / strength 指定時は Edit-I2I へ委譲)。

    src_images は preprocess_image() 済みリスト(0枚=テキストのみ生成)。
    """
    if req.get("strength") is not None:
        return run_edit_i2i(req, src_images)

    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")
    text_only = not src_images

    pipe = edit_mod.get_edit_pipeline(req.get("quant"))

    steps_eff = req["steps"]
    cfg_eff = req["cfg"]
    shift_eff = req.get("shift")
    lightning = req.get("lightning", True)
    lightning_active = edit_mod.set_edit_lightning(pipe, lightning)
    # 直前が charsheet だった場合の angles adapter 残留を必ずリセットする
    # (fp8-lightning 構成では set_edit_lightning() がアダプタ状態を触らないため必須)。
    edit_mod.set_edit_angles(pipe, False)
    lightning_unavailable = lightning and not lightning_active
    if lightning and lightning_active:
        if steps_eff <= 0 or steps_eff > 12:
            steps_eff = 4
        cfg_eff = LIGHTNING_TRUE_CFG_SCALE
        shift_eff = shift_eff if shift_eff is not None else LIGHTNING_SHIFT
    elif lightning_unavailable:
        if steps_eff <= 8:
            steps_eff = 30
        if cfg_eff <= 1.5:
            cfg_eff = 4.0

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    # TE CPU退避(DS_EDIT_TE_OFFLOAD、既定 auto)。GPU復帰は pipe(...) 直前に行うこと
    # (QwenImageEditPlusPipeline は encode_prompt より前に _execution_device を確定する、
    # 元 CLAUDE.md 23番の罠)。GGUF full tier は退避を無効化(60番)。
    gguf_full_tier = state.edit_group.get("gguf_tier") == edit_mod.GGUF_TIER_FULL
    te_offload = should_offload_edit_text_encoder(width, height, gguf_full_tier=gguf_full_tier, pipe=pipe)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    call_kwargs = dict(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback("edit", steps_eff),
    )
    if text_only:
        # 参照画像なし(テキストのみ生成)。ランタイムパッチ済み __call__ を使う
        # (edit_text_only.py 参照。画像ありの通常経路は下の else 節のまま)。
        patched_call = edit_text_only_mod.get_text_only_edit_call()
        result = patched_call(pipe, image=None, **call_kwargs)
    else:
        # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
        result = pipe(image=src_images, **call_kwargs)
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit")
    out_image.save(out_path)

    meta = {
        "mode": "edit",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "text_to_image": text_only,
        "edit_transformer_fallback": state.edit_group["fallback"],
        "quant": state.edit_group["quant"],
        "lightning_requested": lightning,
        "lightning_unavailable_reason": (
            "GGUF量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
        "te_offload": te_offload,
        "gguf_tier": state.edit_group.get("gguf_tier"),
    }
    return meta


def run_edit_i2i(req: dict, src_images: list) -> dict:
    """Edit(2511)transformer を参照共有した I2I(strength 指定時、元 CLAUDE.md 64番)。"""
    if len(src_images) != 1:
        raise ValueError("strength 指定時(Edit-I2I)は参照画像をちょうど1枚アップロードしてください。")
    strength = req.get("strength")
    if strength is None or not (0 < strength <= 1):
        raise ValueError("strength は 0 より大きく 1 以下の範囲で指定してください。")
    src_image = src_images[0]

    pipe = edit_mod.get_edit_i2i_pipeline(req.get("quant"))

    steps_eff = req["steps"]
    cfg_eff = req["cfg"]
    shift_eff = req.get("shift")
    lightning = req.get("lightning", True)
    lightning_active = edit_mod.set_edit_lightning(pipe, lightning)
    edit_mod.set_edit_angles(pipe, False)  # charsheet の angles 残留リセット(run_edit() と同じ理由)
    lightning_unavailable = lightning and not lightning_active
    if lightning and lightning_active:
        if steps_eff <= 0 or steps_eff > 12:
            steps_eff = 4
        cfg_eff = LIGHTNING_TRUE_CFG_SCALE
        shift_eff = shift_eff if shift_eff is not None else LIGHTNING_SHIFT
    elif lightning_unavailable:
        if steps_eff <= 8:
            steps_eff = 30
        if cfg_eff <= 1.5:
            cfg_eff = 4.0

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    gguf_full_tier = state.edit_group.get("gguf_tier") == edit_mod.GGUF_TIER_FULL
    src_w, src_h = src_image.size
    te_offload = should_offload_edit_text_encoder(src_w, src_h, gguf_full_tier=gguf_full_tier, pipe=pipe)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        image=src_image,
        strength=strength,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        generator=generator,
        callback_on_step_end=_progress_callback("edit_i2i", steps_eff),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit_i2i")
    out_image.save(out_path)

    meta = {
        "mode": "edit",
        "edit_i2i": True,
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "strength": strength,
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": 1,
        "text_to_image": False,
        "input_size": list(src_image.size),
        "te_offload": te_offload,
        "edit_transformer_fallback": state.edit_group["fallback"],
        "quant": state.edit_group["quant"],
        "lightning_requested": lightning,
        "lightning_unavailable_reason": (
            "GGUF量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
        "gguf_tier": state.edit_group.get("gguf_tier"),
    }
    return meta


def run_edit_charsheet(req: dict, src_images: list) -> dict:
    """charsheet(新方式): Edit グループの Multiple-angles adapter を有効化して1方向生成する。

    diffusers-server 本体の edit_angles 系3変種の代替(probe_gguf_charsheet.py で
    品質実証済みの設計)。生成パラメータは Lightning 前提(steps=4, cfg=1.0,
    shift=LIGHTNING_SHIFT)。height/width は呼び出し側(apps/charsheet)が明示する前提。
    """
    if not src_images:
        raise ValueError("参照画像を1〜3枚アップロードしてください。")
    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    pipe = edit_mod.get_edit_pipeline(req.get("quant"))

    if not state.edit_group.get("angles_available"):
        reason = state.edit_group.get("angles_unavailable_reason")
        raise RuntimeError(
            f"Multiple-angles LoRA が利用できないため charsheet を生成できません: {reason}"
        )

    # GGUF/bf16 adapter 構成: Lightning を有効化(set_adapters(["lightning"]))してから
    # angles を重ねる(set_adapters(["lightning","angles"]))。fp8-lightning 構成:
    # Lightning は fuse 済みのため set_edit_lightning() は状態を触らず True を返し、
    # set_edit_angles(True) が angles のみ有効化する。
    lightning_active = edit_mod.set_edit_lightning(pipe, True)
    angles_active = edit_mod.set_edit_angles(pipe, True)
    if not angles_active:
        raise RuntimeError("Multiple-angles adapter の有効化に失敗しました(ログ参照)。")

    steps_eff = req["steps"] if req.get("steps") and 0 < req["steps"] <= 12 else 4
    if not lightning_active:
        # GGUF で Lightning LoRA のロードに失敗している場合のフォールバック
        # (angles のみ有効の30steps。品質未検証のため通常は到達しない防御分岐)。
        steps_eff = 30
        cfg_eff = 4.0
        shift_eff = req.get("shift")
    else:
        cfg_eff = LIGHTNING_TRUE_CFG_SCALE
        shift_eff = req.get("shift") if req.get("shift") is not None else LIGHTNING_SHIFT

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    gguf_full_tier = state.edit_group.get("gguf_tier") == edit_mod.GGUF_TIER_FULL
    te_offload = should_offload_edit_text_encoder(width, height, gguf_full_tier=gguf_full_tier, pipe=pipe)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
    result = pipe(
        image=src_images,
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback(
            "charsheet", steps_eff, req.get("_progress_extra")
        ),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit_charsheet")
    out_image.save(out_path)

    meta = {
        "mode": "edit_charsheet",
        "te_offload": te_offload,
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": lightning_active,
        "angles": True,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "quant": state.edit_group["quant"],
        "gguf_tier": state.edit_group.get("gguf_tier"),
    }
    return meta


def run_controlnet(req: dict, ctrl_img: Image.Image) -> dict:
    """ControlNet Union(本サーバでは API 非公開だが family の mode としては残す)。"""
    w = _round16(req["width"]) if req.get("width") else _round16(ctrl_img.width)
    h = _round16(req["height"]) if req.get("height") else _round16(ctrl_img.height)
    if (w, h) != ctrl_img.size:
        ctrl_img = ctrl_img.resize((w, h), Image.LANCZOS)

    pipe = controlnet_mod.get_controlnet_pipeline()
    ensure_text_encoder_on_gpu(pipe)
    generator, used_seed = make_generator(req.get("seed", -1))

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or " ",
        control_image=ctrl_img,
        controlnet_conditioning_scale=req.get("controlnet_conditioning_scale", 1.0),
        width=w,
        height=h,
        num_inference_steps=req["steps"],
        true_cfg_scale=req["cfg"],
        generator=generator,
        callback_on_step_end=_progress_callback("controlnet", req["steps"]),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("controlnet")
    image.save(out_path)

    meta = {
        "mode": "controlnet",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", " "),
        "control_type": req.get("control_type", "canny"),
        "controlnet_conditioning_scale": req.get("controlnet_conditioning_scale", 1.0),
        "width": w,
        "height": h,
        "steps": req["steps"],
        "cfg": req["cfg"],
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    return meta


def run_inpaint(req: dict, src_image: Image.Image, mask_img: Image.Image) -> dict:
    """ControlNet Inpainting(アウトペイントもこの経路を流用する)。

    共有 text_encoder が直前の呼び出しで CPU 退避されたままのケースの保険として
    ensure_text_encoder_on_gpu() を必ず呼ぶ(元 CLAUDE.md 48番)。
    """
    w = _round16(src_image.width)
    h = _round16(src_image.height)
    if (w, h) != src_image.size:
        src_image = src_image.resize((w, h), Image.LANCZOS)
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.NEAREST)

    pipe = controlnet_mod.get_controlnet_inpaint_pipeline()
    ensure_text_encoder_on_gpu(pipe)
    generator, used_seed = make_generator(req.get("seed", -1))

    _reset_vram_stats()
    t0 = time.time()
    # control_image は元画像(全体)、control_mask がマスク(白=編集領域)。
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or " ",
        control_image=src_image,
        control_mask=mask_img,
        controlnet_conditioning_scale=req.get("controlnet_conditioning_scale", 1.0),
        width=w,
        height=h,
        num_inference_steps=req["steps"],
        true_cfg_scale=req["cfg"],
        generator=generator,
        callback_on_step_end=_progress_callback("inpaint", req["steps"]),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("inpaint")
    out_image.save(out_path)

    meta = {
        "mode": "inpaint",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", " "),
        "controlnet_conditioning_scale": req.get("controlnet_conditioning_scale", 1.0),
        "width": w,
        "height": h,
        "steps": req["steps"],
        "cfg": req["cfg"],
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    return meta


def run_canny(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> dict:
    """cv2.Canny によるエッジ検出制御画像の生成(GPU不使用)。"""
    import cv2
    import numpy as np

    src = image.convert("RGB")
    arr = np.array(src)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    out_img = Image.fromarray(edges_rgb)

    out_path, out_name = _new_output_path("canny")
    out_img.save(out_path)
    return {"image_url": f"/outputs/{out_name}"}
