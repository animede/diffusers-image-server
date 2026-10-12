# -*- coding: utf-8 -*-
"""
Iris-3B ファミリー(T2I のみ、4-step Turbo LoRA)。

モデル: speridlabs/iris-3b(Apache 2.0)。VAEを持たない画素空間DiT 3B + Qwen3-VL-4B-Instruct
テキストエンコーダ。pentacoxian-dev/iris-3b-turbo-lora(DMD2蒸留、Apache 2.0)をロード時に
重みへマージし、4ステップ・CFG焼き込み(negative/cfg なし)で生成する。

diffusers のパイプラインではないため、推論コード iris3b(pentaCoxian/iris-3b-turbo)を使う。
venv へ pip install はしない(CLAUDE.md)ため、DS_IRIS_CODE_DIR(既定 ~/iris-3b-turbo)の
チェックアウトの src/ を sys.path に足して import する(iris3b が既に import 可能ならそれを使う)。
iris3b の推論経路が要るのは torch / transformers(Qwen3-VL対応)/ safetensors / omegaconf のみで、
いずれも本 venv(comfy-env 継承)に揃っている(2026-10-12、torch 2.9 / transformers 5.1 で実測)。

排他構成: Qwen 系・JoyAI と同時常駐させない(48GB の GPU1 で Qwen Inpaint 約39GB + Iris 約7.8GB
は余裕がないため)。registry.load() 経由の切替で相手を自動 unload する。

このパッケージを import した時点で core.registry.registry に自動登録される。
"""
from core.registry import registry

from families.iris3b.family import IrisFamily, render_size

__all__ = ["IrisFamily", "family", "render_size"]

family = IrisFamily()
registry.register(family, exclusive_with=["qwen_image", "joyai"])
