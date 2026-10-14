# -*- coding: utf-8 -*-
"""
FLUX.2 klein ファミリー(4B / 9B、T2I・I2I・最大8枚の参照画像 Ref2I、4 steps)。

移植元: /home/animede/flux2-klein-studio(Klein Reference Studio、2026-10-14)の engine.py /
model_config.py。モデル構成・参照画像の縮小規則はそのまま引き継ぐ:
  4B: transformer GGUF Q4_K_M + text_encoder Qwen3-4B NF4(Apache-2.0)
  9B: transformer GGUF Q3_K_M + text_encoder Qwen3-8B NF4(FLUX Non-Commercial License)

移植元との違い:
  - 移植元は 8GB 級 GPU 向けに `torch.cuda.set_per_process_memory_fraction()`(7GiB)+
    `enable_model_cpu_offload()` を使うが、前者は**プロセス全体**に効いて同居する Qwen 系を
    壊すため使わない。オフロードは DS_KLEIN_OFFLOAD で選ぶ(既定 none = 全常駐)。
  - ジョブ方式ではなく本サーバの同期 API(core.gpu.generation_lock + registry.load())に合わせる。

排他構成: Qwen 系・JoyAI・Iris と同時常駐させない(registry.load() 経由の切替で相手を自動 unload)。
4B と 9B もファミリー内で同時には載せない(切替時に解放してから読む、移植元と同じ)。

このパッケージを import した時点で core.registry.registry に自動登録される。
"""
from core.registry import registry

from families.klein.family import (
    DEFAULT_MODEL,
    MAX_REFERENCES,
    MAX_SIDE,
    MIN_SIDE,
    MODELS,
    KleinFamily,
    normalize_model,
    reference_max_side,
)

__all__ = ["KleinFamily", "family", "MODELS", "DEFAULT_MODEL", "MAX_REFERENCES", "MIN_SIDE", "MAX_SIDE",
           "normalize_model", "reference_max_side"]

family = KleinFamily()
registry.register(family, exclusive_with=["qwen_image", "joyai", "iris3b"])
