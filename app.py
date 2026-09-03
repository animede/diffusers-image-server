# -*- coding: utf-8 -*-
"""
diffusers-image-server 本体(FastAPI、薄く保つ)。

diffusers-server(port 8601)から Edit 系機能だけを独立させた軽量サーバ(port 8620)。
提供機能: Edit-2511(指示編集 / テキストのみ生成 / strength I2I)、T2I/I2I(2512・無印)、
JoyAI-Edit-Plus(マルチ参照)、ControlNet Inpainting、アウトペイント、
キャラクターシート(Edit + Multiple-angles adapter 新方式)、LLMプロンプト支援、背景除去。

起動: venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8620
"""
import io
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

import families.qwen_image as qwen_image  # noqa: F401  (import 副作用で registry に登録)
import families.joyai as joyai  # noqa: F401  (import 副作用で registry に登録)
from apps.charsheet import router as charsheet_router
from core import bg as bg_mod
from core import gpu
from core import llm, progress as progress_mod
from core.registry import registry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="DiffusersImageServer")


def _open_upload_image(data: bytes, mode: "str | None" = "RGB") -> Image.Image:
    """アップロード画像を開く共通ヘルパー(EXIF回転の正規化込み)。"""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert(mode) if mode else img


_last_generation = {"mode": None, "elapsed_s": None, "peak_vram_gb": None, "at": None}

MAX_EDIT_IMAGES = qwen_image.MAX_EDIT_IMAGES

_MODE_TO_FAMILY = {
    "t2i": "qwen_image",
    "i2i": "qwen_image",
    "edit": "qwen_image",
    "inpaint": "qwen_image",
    "joyai_edit": "joyai",
}
_JOYAI_MODE_ALIAS = {"joyai_edit": "edit"}


def _record_last_generation(meta: dict) -> None:
    _last_generation.update(
        mode=meta.get("mode"),
        elapsed_s=meta.get("elapsed_s"),
        peak_vram_gb=meta.get("peak_vram_gb"),
        at=datetime.now().isoformat(),
    )


def _generate_or_409(mode: str, request: dict, load_kwargs: "dict | None" = None) -> dict:
    """全ファミリー共通ロック(core.gpu.generation_lock)を非ブロッキングで取得し、
    取得できた場合のみ registry.load() -> family.generate() を呼ぶ(取得失敗は409)。
    registry.load() を必ず先に呼ぶこと(排他グループの自動unload・切替のため。
    元リポジトリ CLAUDE.md 1番の必須パターン)。
    """
    import torch

    from core import gpu

    family_name = _MODE_TO_FAMILY[mode]
    internal_mode = _JOYAI_MODE_ALIAS.get(mode) or mode
    load_kwargs = load_kwargs or {}

    if not gpu.generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="別の生成が実行中です。しばらく待ってから再試行してください。"
        )
    try:
        gpu.empty_cache()
        progress_mod.start_loading(mode)
        registry.load(family_name, internal_mode, **load_kwargs)
        request = {"mode": internal_mode, **request}
        family = registry.get(family_name)
        meta = family.generate(request)
        meta["mode"] = mode
        _record_last_generation(meta)
        return meta
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except torch.cuda.OutOfMemoryError as exc:
        # OOM 後の reserved キャッシュ残留対策(元リポジトリ CLAUDE.md 56番)。
        traceback.print_exc()
        gpu.empty_cache()
        gpu.reset_peak_stats()
        raise HTTPException(
            status_code=500,
            detail=f"{mode}生成中にGPUメモリ不足が発生しました(CUDA OutOfMemory): {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{mode}生成に失敗しました: {exc}")
    finally:
        progress_mod.finish()
        gpu.generation_lock.release()


# ============================================================================
# T2I
# ============================================================================
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = qwen_image.NEGATIVE_PROMPT_DEFAULT
    width: int = 1328
    height: int = 1328
    steps: int = 30
    cfg: float = 4.0
    seed: int = -1
    lightning: bool = False
    shift: Optional[float] = None
    model: Optional[str] = None  # "2512"(既定)| "qwen-image"
    quant: Optional[str] = None  # "gguf-q4_k_m" 等 / "fp8-lightning" / "none"(未指定=現状維持)


@app.post("/api/t2i")
def generate_t2i(req: T2IRequest):
    payload = req.model_dump()
    quant = qwen_image.normalize_quant_request(payload.get("quant"))
    payload["quant"] = quant
    return _generate_or_409(
        "t2i", payload, load_kwargs={"model": payload.get("model"), "quant": quant}
    )


# ============================================================================
# I2I
# ============================================================================
@app.post("/api/i2i")
async def generate_i2i(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(qwen_image.NEGATIVE_PROMPT_DEFAULT),
    strength: float = Form(0.6),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
    lightning: bool = Form(False),
    shift: Optional[float] = Form(None),
    model: Optional[str] = Form(None),
    quant: Optional[str] = Form(None),
):
    contents = await image.read()
    try:
        src_image = _open_upload_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
    src_image = qwen_image.preprocess_image(src_image)

    quant = qwen_image.normalize_quant_request(quant)
    return _generate_or_409(
        "i2i",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "lightning": lightning,
            "shift": shift,
            "model": model,
            "quant": quant,
            "_image": src_image,
        },
        load_kwargs={"model": model, "quant": quant},
    )


# ============================================================================
# Edit(指示編集 / テキストのみ生成(images 0枚) / strength指定で Edit-I2I)
# ============================================================================
@app.post("/api/edit")
async def generate_edit(
    images: List[UploadFile] = File(default=[]),
    prompt: str = Form(...),
    negative_prompt: str = Form(qwen_image.NEGATIVE_PROMPT_DEFAULT),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
    lightning: bool = Form(True),
    shift: Optional[float] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    strength: Optional[float] = Form(None),
    quant: Optional[str] = Form(None),
):
    images = images or []

    src_images = []
    for f in images:
        contents = await f.read()
        # ブラウザはファイル未選択でも空パートを送るため「画像なし」として除外
        # (テキストのみ生成の判定を壊さない)。
        if not contents and not (f.filename or "").strip():
            continue
        try:
            img = _open_upload_image(contents)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
        src_images.append(qwen_image.preprocess_image(img))
    if len(src_images) > MAX_EDIT_IMAGES:
        raise HTTPException(status_code=400, detail=f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    if strength is not None:
        if len(src_images) != 1:
            raise HTTPException(
                status_code=400,
                detail="strength 指定時(Edit-I2I)は参照画像をちょうど1枚アップロードしてください。",
            )
        if not (0 < strength <= 1):
            raise HTTPException(
                status_code=400, detail="strength は 0 より大きく 1 以下の範囲で指定してください。"
            )

    quant = qwen_image.normalize_quant_request(quant)
    return _generate_or_409(
        "edit",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "lightning": lightning,
            "shift": shift,
            "width": width,
            "height": height,
            "strength": strength,
            "quant": quant,
            "_images": src_images,
        },
        load_kwargs={"quant": quant},
    )


# ============================================================================
# Inpaint(ControlNet Inpainting、無印 T2I グループ依存)
# ============================================================================
@app.post("/api/inpaint")
async def generate_inpaint(
    image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    controlnet_conditioning_scale: float = Form(1.0),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
):
    try:
        src_contents = await image.read()
        src_image = _open_upload_image(src_contents)
        mask_contents = await mask_image.read()
        mask_img = _open_upload_image(mask_contents, "L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "inpaint",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "_image": src_image,
            "_mask_image": mask_img,
        },
    )


# ============================================================================
# アウトペイント(インペイント機構の流用による画角拡張。元リポジトリ CLAUDE.md 54番/56番)
# ============================================================================
@app.post("/api/outpaint")
async def generate_outpaint(
    image: UploadFile = File(...),
    width: int = Form(1280),
    height: int = Form(720),
    prompt: str = Form(""),
    negative_prompt: str = Form(" "),
    seed: int = Form(-1),
    feather: int = Form(64),
    steps: int = Form(30),
    cfg: float = Form(4.0),
):
    """アウトペイント: 入力画像を中央配置し、余白帯をインペイントで生成して拡張する。
    engine は qwen(ControlNet Inpainting)のみ(z-image は本サーバ非搭載)。
    実行前に全ファミリーを解放する保守的な安全策(元リポジトリ CLAUDE.md 56番)を維持。
    """
    from apps import outpaint as outpaint_mod
    from core import gpu

    if not gpu.generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="別の生成が実行中です。しばらく待ってから再試行してください。"
        )
    try:
        registry.unload(None)
    finally:
        gpu.generation_lock.release()

    try:
        contents = await image.read()
        src_image = _open_upload_image(contents)
        canvas, mask, paste_box = outpaint_mod.build_outpaint_canvas(
            src_image, width, height, feather
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    meta = _generate_or_409(
        "inpaint",
        {
            "prompt": prompt.strip() or " ",
            "negative_prompt": negative_prompt,
            "controlnet_conditioning_scale": 1.0,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "_image": canvas,
            "_mask_image": mask,
        },
    )

    generated_name = str(meta.get("image_url") or "").rsplit("/", 1)[-1]
    generated_path = os.path.join(OUTPUTS_DIR, generated_name)
    try:
        generated = Image.open(generated_path)
        final = outpaint_mod.recomposite_center(
            generated, src_image, paste_box, feather, width, height
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"outpaint_{ts}_{uuid.uuid4().hex[:8]}.png"
        final.save(os.path.join(OUTPUTS_DIR, out_name))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"中央再合成に失敗しました: {exc}")

    meta.update(
        mode="outpaint",
        image_url=f"/outputs/{out_name}",
        raw_inpaint_url=f"/outputs/{generated_name}",
        engine="qwen",
        feather=int(feather),
        width=final.width,
        height=final.height,
        paste_box=list(paste_box),
    )
    return meta


# ============================================================================
# Canny(おまけ機能、GPU不使用。ロック不要)
# ============================================================================
@app.post("/api/canny")
async def make_canny(
    image: UploadFile = File(...),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200),
):
    contents = await image.read()
    try:
        src = _open_upload_image(contents, None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
    try:
        return qwen_image.run_canny(src, low_threshold, high_threshold)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"opencv(cv2)が利用できません: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Canny生成に失敗しました: {exc}")


# ============================================================================
# 背景削除ユーティリティ(CPU/ONNX、generation_lock 不使用)
# ============================================================================
@app.post("/api/remove_bg")
async def api_remove_bg(
    image: UploadFile = File(...),
    method: str = Form(bg_mod.DEFAULT_BG_METHOD),
):
    contents = await image.read()
    try:
        src = _open_upload_image(contents, "RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    used = bg_mod.resolve_method(method)
    t0 = time.time()
    try:
        out_image = bg_mod.remove_background(src, method=used)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"背景削除に失敗しました: {exc}")
    elapsed = time.time() - t0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"removebg_{ts}_{uuid.uuid4().hex[:8]}.png"
    out_image.save(os.path.join(OUTPUTS_DIR, out_name))
    return {"mode": "remove_bg", "image_url": f"/outputs/{out_name}",
            "elapsed_s": elapsed, "method": used}


# ============================================================================
# JoyAI-Image-Edit-Plus(複数参照画像編集)
# ============================================================================
@app.post("/api/joyai/edit")
async def generate_joyai_edit(
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(joyai.DEFAULT_STEPS),
    guidance_scale: float = Form(joyai.DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    quant: Optional[str] = Form(None),
):
    if not images:
        raise HTTPException(status_code=400, detail="参照画像を1〜3枚アップロードしてください。")
    if len(images) > joyai.MAX_REF_IMAGES:
        raise HTTPException(status_code=400, detail=f"参照画像は最大{joyai.MAX_REF_IMAGES}枚までです。")

    src_images = []
    for f in images:
        contents = await f.read()
        try:
            img = _open_upload_image(contents)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
        src_images.append(img)

    quant = joyai.normalize_quant_request(quant)
    return _generate_or_409(
        "joyai_edit",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height,
            "quant": quant,
            "_images": src_images,
        },
        load_kwargs={"quant": quant},
    )


# ============================================================================
# unload / status / progress
# ============================================================================
class UnloadRequest(BaseModel):
    # "t2i" | "edit" | "controlnet" | "joyai" | "all"
    target: str = "all"


@app.post("/api/unload")
def unload(req: UnloadRequest):
    if req.target == "joyai":
        result = {"joyai": joyai.family.unload()}
    elif req.target not in ("t2i", "edit", "controlnet", "all"):
        raise HTTPException(
            status_code=400,
            detail="target は t2i / edit / controlnet / joyai / all のいずれかです",
        )
    elif req.target == "all":
        result = registry.unload(None)
    else:
        result = qwen_image.family.unload_target(req.target)

    # 解放するモデルが無かった場合でも、PyTorch のキャッシュは必ず返却する。
    # ここを条件付きにすると「何もロードされていないのに reserved が数GB残る」状態に
    # なる(実測: freed=[] のとき allocated 0.13GB / reserved 2.34GB が居座り、
    # nvidia-smi では 2.8GB を占有していた)。
    # 画像サーバと動画サーバが同一GPUを共有する構成では、この数GBが動画側の
    # OOM に直結するため、unload は「必ずキャッシュを返す」契約にしておく。
    def _reserved() -> float:
        snap = gpu.vram_snapshot() or {}
        return float(snap.get("reserved_gb") or 0.0)

    before = _reserved()
    gpu.empty_cache()
    after = _reserved()
    if before - after > 0.05:
        print(f"[unload] released cached VRAM: {before:.2f}GB -> {after:.2f}GB")
    return result


@app.get("/api/progress")
def get_progress():
    return progress_mod.snapshot()


@app.get("/api/status")
def status():
    st = registry.status_all()
    st["last_generation"] = _last_generation
    # 旧クライアント互換: qwen_image 固有のキーをトップレベルにも展開する。
    qwen_status = st.get("qwen_image", {})
    merged = dict(qwen_status)
    merged.update(st)
    merged.pop("qwen_image", None)
    return merged


# ============================================================================
# LLMプロンプト支援(非GPU・排他不要)
# ============================================================================
class PromptLLMRequest(BaseModel):
    text: str


@app.post("/api/prompt/enhance")
def prompt_enhance(req: PromptLLMRequest):
    start = time.time()
    try:
        result = llm.enhance_prompt(req.text)
    except llm.LLMConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"result": result, "elapsed_s": time.time() - start}


@app.post("/api/prompt/translate")
def prompt_translate(req: PromptLLMRequest):
    start = time.time()
    try:
        result = llm.translate_prompt(req.text)
    except llm.LLMConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"result": result, "elapsed_s": time.time() - start}


# ============================================================================
# charsheet(ジョブ式API)
# ============================================================================
app.include_router(charsheet_router, prefix="/api/charsheet")


# ============================================================================
# 生成画像・静的ファイル配信
# ============================================================================
class NoCacheStaticFiles(StaticFiles):
    """静的ファイルにブラウザキャッシュを効かせない(開発中の反映漏れ防止)。"""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
