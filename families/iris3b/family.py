# -*- coding: utf-8 -*-
"""
IrisFamily: core.registry.ModelFamily の実装(Iris-3B T2I、Turbo LoRA マージ済み)。

環境変数:
  DS_IRIS_CODE_DIR    iris-3b-turbo チェックアウト(既定 ~/iris-3b-turbo)
  DS_IRIS_CHECKPOINT  HF ID または model.safetensors+config.yaml のディレクトリ(既定 speridlabs/iris-3b)
  DS_IRIS_LORA        Turbo LoRA。owner/repo[:name] またはローカル .safetensors
                      (既定 pentacoxian-dev/iris-3b-turbo-lora)
  DS_IRIS_QUANT       int8(既定。DiT幹とテキストエンコーダを重みint8、常駐約7.2GB)| none(bf16、約13GB)
"""
import gc
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime

from core import gpu
from core import progress as progress_mod
from core.registry import ModelFamily

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs"
)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Turbo LoRA の蒸留解像度(1024²)付近。auto_size 指定時はこの画素数に合わせる。
TARGET_AREA = 1024 * 1024
MIN_SIDE, MAX_SIDE = 256, 2048

_lock = threading.Lock()
_state = {"loaded": False, "model": None, "text_encoder": None, "meta": None,
          "quant": None, "load_time_s": None}


def render_size(width: int, height: int, auto_size: bool = False) -> "tuple[int, int]":
    """生成サイズ。16px パッチの倍数に丸める。auto_size なら縦横比を保ったまま約1MPにする
    (Momo Song の /generate/ は 896x512 等の表示サイズを送ってくるため)。
    """
    if auto_size:
        ratio = width / height
        ideal_h = math.sqrt(TARGET_AREA / ratio)
        best = None
        for h in range(max(16, int(ideal_h * 0.9) // 16 * 16), int(ideal_h * 1.1) + 16, 16):
            w = max(16, round(h * ratio / 16) * 16)
            key = (abs(w / h - ratio), abs(w * h - TARGET_AREA))
            if best is None or key < best[0]:
                best = (key, (w, h))
        return best[1]
    clamp = lambda v: min(MAX_SIDE, max(MIN_SIDE, round(int(v) / 16) * 16))  # noqa: E731
    return clamp(width), clamp(height)


def _import_iris():
    code_dir = os.environ.get("DS_IRIS_CODE_DIR", os.path.expanduser("~/iris-3b-turbo"))
    src = os.path.join(code_dir, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import iris3b.text  # noqa: F401  (Qwen3-VL エンコーダを登録)
        from iris3b import config, lora, registry as iris_registry, sampling  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"iris3b を import できません。DS_IRIS_CODE_DIR({code_dir})に "
            "https://github.com/pentaCoxian/iris-3b-turbo をチェックアウトしてください"
        ) from exc


def _load() -> None:
    import torch

    _import_iris()
    from huggingface_hub import snapshot_download
    from iris3b.config import inference_config
    from iris3b.lora import resolve_adapter
    from iris3b.registry import TEXT_ENCODERS
    from iris3b.sampling import load_model_streaming, read_config

    t0 = time.time()
    checkpoint = os.environ.get("DS_IRIS_CHECKPOINT", "speridlabs/iris-3b")
    if not os.path.isdir(checkpoint):
        # リポジトリには depth / upscaler の派生版も同梱されているが、T2I 本体だけを取る
        checkpoint = snapshot_download(checkpoint, allow_patterns=["model.safetensors", "config.yaml"])
    adapter = resolve_adapter(os.environ.get("DS_IRIS_LORA", "pentacoxian-dev/iris-3b-turbo-lora"))
    quant = os.environ.get("DS_IRIS_QUANT", "int8").strip().lower()
    if quant not in ("int8", "none"):
        raise ValueError("DS_IRIS_QUANT は int8 / none のいずれかです")

    device = torch.device("cuda")
    cfg = inference_config(read_config(checkpoint), [])
    if quant == "int8":
        cfg.text_encoder.quantize = "int8"
    text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)
    # 画素ステージと patch/timestep 埋め込みは小さいので int8 でも bf16 のまま残す
    small = ("pixel_blocks", "pixel_embedder", "final_layer", "s_embedder", "t_embedder")
    keep = (lambda name: name.startswith(small)) if quant == "int8" else None
    model, meta = load_model_streaming(
        cfg.model, checkpoint, device, torch.bfloat16, keep, str(adapter), 1.0
    )
    model.eval()
    if meta.get("turbo_steps") is None:
        raise ValueError(f"{adapter} は Turbo アダプタではありません(turbo_steps メタデータなし)")
    _state.update(loaded=True, model=model, text_encoder=text_encoder, meta=meta,
                  quant=quant, load_time_s=round(time.time() - t0, 2))
    print(f"[families.iris3b] loaded (quant={quant}) in {_state['load_time_s']}s")


def _generate(req: dict) -> dict:
    import torch
    from iris3b.devices import inference_autocast
    from iris3b.distill import generate_turbo

    width, height = render_size(int(req.get("width") or 1344), int(req.get("height") or 768),
                                bool(req.get("auto_size")))
    seed = req.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int(torch.seed() % (2 ** 31))
    meta = _state["meta"]
    steps = int(meta["turbo_steps"])

    torch.cuda.reset_peak_memory_stats()
    progress_mod.start_generating("iris_t2i", 0)  # generate_turbo はステップ callback 非対応
    t0 = time.time()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with inference_autocast(torch.device("cuda")):
        image = generate_turbo(
            _state["model"], _state["text_encoder"], [req["prompt"]], height=height, width=width,
            steps=steps, shift=float(meta["shift"]), generator=generator, device=torch.device("cuda"),
        )
    pixels = ((image[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    array = pixels.permute(1, 2, 0).cpu().numpy()
    elapsed = time.time() - t0
    progress_mod.set_phase("saving")

    from PIL import Image

    name = f"iris_t2i_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    Image.fromarray(array, "RGB").save(os.path.join(OUTPUTS_DIR, name))
    return {
        "mode": "iris_t2i",
        "model": "iris-3b-turbo",
        "prompt": req["prompt"],
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "quant": _state["quant"],
        "elapsed_s": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "image_url": f"/outputs/{name}",
    }


class IrisFamily(ModelFamily):
    name = "iris3b"

    def load(self, mode: str, **kwargs) -> None:
        if mode != "t2i":
            raise ValueError(f"unknown iris3b mode: {mode!r}")
        with _lock:
            if not _state["loaded"]:
                _load()

    def generate(self, request: dict) -> dict:
        if request["mode"] != "t2i":
            raise ValueError(f"unknown iris3b mode: {request['mode']!r}")
        if not request.get("prompt", "").strip():
            raise ValueError("prompt が空です")
        return _generate(request)

    def unload(self) -> dict:
        freed = []
        with _lock:
            if _state["loaded"]:
                _state.update(loaded=False, model=None, text_encoder=None, meta=None,
                              quant=None, load_time_s=None)
                freed.append("iris3b")
            gc.collect()
            gpu.empty_cache()
            gpu.reset_peak_stats()
        print(f"[families.iris3b] unloaded: {freed}")
        return {"freed": freed}

    def status(self) -> dict:
        return {
            "loaded": _state["loaded"],
            "quant": _state["quant"],
            "load_time_s": _state["load_time_s"],
        }
