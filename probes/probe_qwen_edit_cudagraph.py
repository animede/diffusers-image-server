"""qwen Edit(2511 fp8-lightning)への CUDA Graph 適用可否 probe。

判定項目:
  1. capture が通るか(fp8 layerwise casting フック・rope 計算等の capture 互換)
  2. 出力が eager と bit 一致するか
  3. replay を繰り返しても出力が劣化しないか(fp8 フックが param.data を
     張り替える実装だと、capture が古いアドレスを焼き込み重みが壊れる懸念)
  4. 速度(eager vs graph、pipe() 全体の壁時計)
"""
import importlib.util
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("DS_QUANT", "fp8-lightning")

import numpy as np
import torch
from PIL import Image

# LTX 側で実証済みの runner を流用(純 torch 実装)
spec = importlib.util.spec_from_file_location(
    "cudagraph", "/home/animede/diffusers-movie-server/backends/ltx2_5/app/cudagraph.py")
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)

t0 = time.time()
from families.qwen_image import edit as edit_mod

pipe = edit_mod.get_edit_pipeline()
print(f"pipeline loaded: {time.time()-t0:.1f}s", flush=True)
try:
    edit_mod.set_scheduler_shift(pipe.scheduler, edit_mod.LIGHTNING_SHIFT)
    print("scheduler shift set:", edit_mod.LIGHTNING_SHIFT, flush=True)
except Exception as e:
    print("shift set skipped:", e, flush=True)

img = Image.open("/home/animede/minimax-h3/outputs/ab_ref2va/input_image_person.png").convert("RGB")
PROMPT = "change the jacket color to red, keep everything else the same"


def gen(seed=12345):
    g = torch.Generator("cuda").manual_seed(seed)
    t1 = time.time()
    with torch.no_grad():
        out = pipe(image=[img], prompt=PROMPT, height=1024, width=1024,
                   num_inference_steps=4, true_cfg_scale=1.0, generator=g)
    dt = time.time() - t1
    return np.asarray(out.images[0]), dt


def cmp(label, a, b):
    eq = np.array_equal(a, b)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    print(f"{label}: equal={eq} max|d|={d.max()} mean|d|={d.mean():.4f}", flush=True)
    return eq


# eager: warm + 基準2本(自己 bit 安定性の確認)
_, dtw = gen()
A1, dt1 = gen()
A2, dt2 = gen()
print(f"eager: warm {dtw:.2f}s / steady {dt1:.2f}s, {dt2:.2f}s", flush=True)
cmp("eager self-consistency (A1 vs A2)", A1, A2)

# graph 化
runner = cg.ForwardGraphRunner(pipe.transformer, max_captures=4)
runner.install()
G = []
times = []
for i in range(5):
    gi, ti = gen()
    G.append(gi)
    times.append(ti)
print("graph gens:", " ".join(f"{t:.2f}s" for t in times),
      f"| replays={runner.replays}", flush=True)
ok1 = cmp("graph vs eager (G1 vs A1)", G[0], A1)
ok2 = cmp("graph run5 vs run1 (劣化検査)", G[4], G[0])
ok3 = cmp("graph run5 vs eager", G[4], A1)
print("VERDICT:",
      "PASS(bit一致・劣化なし)" if (ok1 and ok2) else
      ("DRIFT(劣化あり=fp8フック非互換の疑い)" if ok1 and not ok2 else
       ("STABLE-BUT-DIFFERENT" if ok2 else "FAIL")), flush=True)
print(f"speed: eager {min(dt1,dt2):.2f}s -> graph steady {min(times[1:]):.2f}s", flush=True)
