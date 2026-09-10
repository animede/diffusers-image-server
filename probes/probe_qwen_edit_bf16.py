"""DS_QUANT=none(bf16 + Lightning adapter)での Edit 速度測定(fp8 キャスト排除の効果)。"""
import os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ["DS_QUANT"] = "none"
import numpy as np, torch
from PIL import Image
from families.qwen_image import edit as edit_mod
t0=time.time(); pipe = edit_mod.get_edit_pipeline(); print(f"loaded {time.time()-t0:.1f}s", flush=True)
try:
    edit_mod.set_edit_lightning(pipe, True); print("lightning adapter ON", flush=True)
except Exception as e: print("lightning:", e, flush=True)
img = Image.open("/home/animede/minimax-h3/outputs/ab_ref2va/input_image_person.png").convert("RGB")
def gen(steps=4):
    g=torch.Generator("cuda").manual_seed(12345); t1=time.time()
    with torch.no_grad():
        pipe(image=[img], prompt="change the jacket color to red, keep everything else the same",
             height=1024, width=1024, num_inference_steps=steps, true_cfg_scale=1.0, generator=g)
    return time.time()-t1
print(f"warm {gen():.2f}s", flush=True)
print(f"steady4 {gen():.2f}s {gen():.2f}s | steps1 {gen(1):.2f}s", flush=True)
print(f"vram peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
