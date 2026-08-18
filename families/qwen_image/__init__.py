# -*- coding: utf-8 -*-
"""
Qwen-Image 系ファミリー(diffusers-image-server 版: T2I / I2I / Edit / Edit-charsheet /
Inpaint)。

diffusers-server の families/qwen_image を刈り込んで移植したパッケージ
(layered / edit_angles 系3変種を削除。charsheet は edit グループ +
Multiple-angles adapter トグルで実現する)。

このパッケージを import した時点で core.registry.registry にシングルトンとして自動登録する
(app.py 起動時に import families.qwen_image するだけでよい)。
"""
from core.registry import registry

from families.qwen_image.family import QwenImageFamily
from families.qwen_image.generate import (
    MAX_EDIT_IMAGES,
    OUTPUTS_DIR,
    make_generator,
    preprocess_image,
    run_canny,
    run_controlnet,
    run_edit,
    run_edit_charsheet,
    run_i2i,
    run_inpaint,
    run_t2i,
    t2i_model_meta,
)
from families.qwen_image.lifecycle import get_status, unload
from families.qwen_image.paths import NEGATIVE_PROMPT_DEFAULT, is_2512_model
from families.qwen_image.runtime import normalize_quant_request
from families.qwen_image.state import get_runtime_config, lock

__all__ = [
    "QwenImageFamily",
    "family",
    "get_status",
    "unload",
    "run_t2i",
    "run_i2i",
    "run_edit",
    "run_edit_charsheet",
    "run_controlnet",
    "run_inpaint",
    "run_canny",
    "preprocess_image",
    "make_generator",
    "t2i_model_meta",
    "MAX_EDIT_IMAGES",
    "OUTPUTS_DIR",
    "NEGATIVE_PROMPT_DEFAULT",
    "is_2512_model",
    "get_runtime_config",
    "lock",
    "normalize_quant_request",
]

# プロセス内シングルトン(core.registry.registry に登録)。
family = QwenImageFamily()
registry.register(family)
