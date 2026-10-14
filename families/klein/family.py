# -*- coding: utf-8 -*-
"""
KleinFamily: core.registry.ModelFamily の実装(FLUX.2 klein 4B / 9B)。

環境変数:
  DS_KLEIN_MODELS_DIR  モデル置き場(既定 ~/flux2-klein-studio/models)。<dir>/klein-4b・
                       <dir>/klein-9b(diffusers 形式、transformer は config のみ)と
                       <dir>/gguf/*.gguf を参照する。無ければ固定リビジョンを HF キャッシュへ取得。
  DS_KLEIN_OFFLOAD     none(既定、全常駐)| model_cpu(移植元と同じ CPU オフロード、低速・省VRAM)
"""
import gc
import math
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from core import gpu
from core import progress as progress_mod
from core.registry import ModelFamily

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs"
)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

STEPS = 4
GUIDANCE = 1.0
MAX_REFERENCES = 8
REFERENCE_MAX_SIDE = 768
MIN_SIDE, MAX_SIDE = 256, 2048

# 移植元 model_config.py と同じ(リビジョン固定・参照画像の画素予算も同値)。
MODELS = {
    "4b": {
        "name": "FLUX.2 klein 4B",
        "license": "Apache-2.0",
        "dir_name": "klein-4b",
        "base_repo": "black-forest-labs/FLUX.2-klein-4B",
        "base_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "gguf_repo": "unsloth/FLUX.2-klein-4B-GGUF",
        "gguf_revision": "0084d1df98e2e2137fe776d55170bc4792ec1d66",
        "gguf_file": "flux-2-klein-4b-Q4_K_M.gguf",
        "quantization": "GGUF Q4_K_M + TE NF4",
        "reference_pixel_budget": 4 * 768 ** 2,
    },
    "9b": {
        "name": "FLUX.2 klein 9B",
        "license": "FLUX Non-Commercial License",
        "dir_name": "klein-9b",
        "base_repo": "unsloth/FLUX.2-klein-9B",
        "base_revision": "455f1273c2f08f1fc93d725c2e6f1d7fc681cd94",
        "gguf_repo": "unsloth/FLUX.2-klein-9B-GGUF",
        "gguf_revision": "fde8634245fe6b749a221c25b34672b5b8fbd079",
        "gguf_file": "flux-2-klein-9b-Q3_K_M.gguf",
        "quantization": "GGUF Q3_K_M + TE NF4",
        "reference_pixel_budget": 2 * 768 ** 2,
    },
}
DEFAULT_MODEL = "4b"
_BASE_PATTERNS = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*",
                  "vae/*", "transformer/config.json", "LICENSE*"]

_lock = threading.Lock()
_state = {"pipe": None, "model": None, "offload": None, "load_time_s": None}


def normalize_model(value) -> str:
    key = str(value or DEFAULT_MODEL).strip().lower()
    if key not in MODELS:
        raise ValueError(f"model は {' / '.join(MODELS)} のいずれかです(指定: {value!r})")
    return key


def reference_max_side(count: int, model_key: str = DEFAULT_MODEL) -> int:
    """参照画像1枚あたりの最大辺(移植元と同じ規則。枚数が増えるほど小さくする)。"""
    if count <= 0:
        return REFERENCE_MAX_SIDE
    bounded = min(REFERENCE_MAX_SIDE, math.sqrt(MODELS[model_key]["reference_pixel_budget"] / count))
    return max(256, int(bounded) // 32 * 32)


def _resolve_files(cfg: dict) -> "tuple[Path, Path]":
    """(diffusers 形式のモデルディレクトリ, GGUF ファイル)。ローカルに無ければ HF から取る。"""
    root = Path(os.environ.get("DS_KLEIN_MODELS_DIR", os.path.expanduser("~/flux2-klein-studio/models")))
    model_dir = root / cfg["dir_name"]
    gguf_path = root / "gguf" / cfg["gguf_file"]
    if not (model_dir / "model_index.json").is_file():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(cfg["base_repo"], revision=cfg["base_revision"],
                                           allow_patterns=_BASE_PATTERNS))
    if not gguf_path.is_file():
        from huggingface_hub import hf_hub_download

        gguf_path = Path(hf_hub_download(cfg["gguf_repo"], cfg["gguf_file"], revision=cfg["gguf_revision"]))
    return model_dir, gguf_path


def _unload_locked() -> bool:
    if _state["pipe"] is None:
        return False
    try:
        _state["pipe"].remove_all_hooks()
    except Exception:  # noqa: BLE001 - フック未登録(none モード)でも落とさない
        pass
    _state.update(pipe=None, model=None, offload=None, load_time_s=None)
    gc.collect()
    gpu.empty_cache()
    return True


def _load_locked(model_key: str) -> None:
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
    from transformers import BitsAndBytesConfig, Qwen3ForCausalLM

    cfg = MODELS[model_key]
    offload = os.environ.get("DS_KLEIN_OFFLOAD", "none").strip().lower()
    if offload not in ("none", "model_cpu"):
        raise ValueError("DS_KLEIN_OFFLOAD は none / model_cpu のいずれかです")
    t0 = time.time()
    model_dir, gguf_path = _resolve_files(cfg)
    device = "cuda"
    dtype = torch.bfloat16
    try:
        # bnb 4bit は .to(dtype) できないので、最初から GPU へ直接ロードする(元 CLAUDE.md 42番)。
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_dir / "text_encoder", local_files_only=True, dtype=dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            ),
            device_map={"": device},
        )
        if offload == "model_cpu":
            text_encoder.to("cpu")
            gc.collect()
            gpu.empty_cache()
        transformer = Flux2Transformer2DModel.from_single_file(
            str(gguf_path), config=str(model_dir), subfolder="transformer",
            local_files_only=True, torch_dtype=dtype,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            model_dir, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=dtype, local_files_only=True,
        )
        pipe.vae.enable_tiling()  # 移植元と同じ(大きい出力での VAE decode ピーク対策)
        if offload == "model_cpu":
            pipe.enable_model_cpu_offload()
        else:
            # pipe.to() は bnb 量子化の text_encoder にも .to() を掛けるので使わず、
            # 残りの部品だけ個別に GPU へ載せる(text_encoder はロード時点で GPU 上)。
            pipe.transformer.to(device)
            pipe.vae.to(device)
        progress_mod.disable_diffusers_tqdm(pipe)
    except Exception:
        gc.collect()
        gpu.empty_cache()
        raise
    _state.update(pipe=pipe, model=model_key, offload=offload, load_time_s=round(time.time() - t0, 2))
    print(f"[families.klein] loaded {cfg['name']} (offload={offload}) in {_state['load_time_s']}s")


def _generate(req: dict) -> dict:
    import torch

    model_key = _state["model"]
    cfg = MODELS[model_key]
    width, height = int(req.get("width") or 1024), int(req.get("height") or 1024)
    for side in (width, height):
        if side < MIN_SIDE or side > MAX_SIDE or side % 32:
            raise ValueError(f"幅・高さは {MIN_SIDE}〜{MAX_SIDE} の32の倍数で指定してください")
    references = list(req.get("_images") or [])
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"参照画像は {MAX_REFERENCES} 枚までです")
    seed = req.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int(torch.seed() % (2 ** 32))
    mode = "klein_ref2i" if references else "klein_t2i"

    torch.cuda.reset_peak_memory_stats()
    progress_mod.start_generating(mode, STEPS)

    def _on_step(pipe, step, timestep, kwargs):
        progress_mod.update_step(step + 1, STEPS)
        return kwargs

    t0 = time.time()
    with torch.inference_mode():
        image = _state["pipe"](
            image=references or None, prompt=req["prompt"], width=width, height=height,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            generator=torch.Generator("cpu").manual_seed(seed),
            max_sequence_length=512, callback_on_step_end=_on_step,
        ).images[0]
    elapsed = time.time() - t0
    progress_mod.set_phase("saving")

    name = f"klein{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    image.save(os.path.join(OUTPUTS_DIR, name))
    return {
        "model": f"flux2-klein-{model_key}",
        "model_name": cfg["name"],
        "license": cfg["license"],
        "quant": cfg["quantization"],
        "offload": _state["offload"],
        "prompt": req["prompt"],
        "width": width,
        "height": height,
        "steps": STEPS,
        "seed": seed,
        "reference_count": len(references),
        "reference_max_side": req.get("_reference_max_side") if references else None,
        "elapsed_s": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "image_url": f"/outputs/{name}",
    }


class KleinFamily(ModelFamily):
    name = "klein"

    def load(self, mode: str, **kwargs) -> None:
        if mode != "generate":
            raise ValueError(f"unknown klein mode: {mode!r}")
        model_key = normalize_model(kwargs.get("model"))
        with _lock:
            if _state["pipe"] is not None and _state["model"] == model_key:
                return
            if _state["pipe"] is not None:
                # 4B と 9B は同時に載せない(移植元と同じ。解放してから読む)
                print(f"[families.klein] switching {_state['model']} -> {model_key}")
                _unload_locked()
            _load_locked(model_key)

    def generate(self, request: dict) -> dict:
        if request["mode"] != "generate":
            raise ValueError(f"unknown klein mode: {request['mode']!r}")
        if not str(request.get("prompt", "")).strip():
            raise ValueError("prompt が空です")
        if _state["pipe"] is None:
            raise RuntimeError("klein モデルがロードされていません")
        return _generate(request)

    def unload(self) -> dict:
        with _lock:
            freed = ["klein"] if _unload_locked() else []
            gpu.reset_peak_stats()
        print(f"[families.klein] unloaded: {freed}")
        return {"freed": freed}

    def status(self) -> dict:
        return {
            "loaded": _state["pipe"] is not None,
            "model": _state["model"],
            "offload": _state["offload"],
            "load_time_s": _state["load_time_s"],
        }
