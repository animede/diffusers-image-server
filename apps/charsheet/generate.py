# -*- coding: utf-8 -*-
"""
charsheet の生成呼び出し(diffusers-image-server 版・新方式)。

diffusers-server 本体の DS_CHARSHEET_METHOD 4方式(bf16-group / bf16-adapters /
fp8-fuse / prompt-only)は本リポジトリには移植しない。代わりに
「Edit グループ(fp8-lightning または GGUF)+ fal 2511 Multiple-Angles LoRA を
transformer-level アダプタとしてロードし、リクエスト単位でトグルする」1方式のみを
提供する(families/qwen_image/edit.py の _apply_angles_adapter() / set_edit_angles()、
diffusers-server 本体の scratchpad probe_gguf_charsheet.py で品質実証済み:
GGUF Q4_K_M で load 20.7s・7.2s/方向・peak 38.83GB(96GB機)、品質は bf16-group
ベースラインと同等と判定済み)。

利点:
  - charsheet 専用の bf16 transformer(~38GB)を別途ロードしない(Edit グループを共有)。
  - 通常 Edit ⇔ charsheet の切替がアダプタトグルのみ(モデル再ロード無し)。
  - GGUF 構成なら 24GB 級カードでも同一経路で動く(見込み。実測は検証項目)。

ジョブ機構(apps/charsheet/jobs.py)からの呼び出しインターフェース
(preprocess_image / ensure_edit_loaded / generate_view / acquire_generation_lock /
release_generation_lock / get_load_info / use_angles_lora)は旧実装と互換。
"""
import os
from typing import Optional

from PIL import Image

from core import gpu
from core.registry import registry

import families.qwen_image as qwen_image

# 生成解像度。環境変数 CHARSHEET_EDIT_SIZE で上書き可能(既定1024、probe と同一)。
_DEFAULT_EDIT_SIZE = 1024


def use_angles_lora() -> bool:
    """本リポジトリの charsheet は常に Multiple-angles LoRA 併用
    (apps/charsheet/prompts.py が VIEWS_LORA(構造化タグ)を選択するための判定)。"""
    return True


def _edit_size() -> int:
    try:
        return int(os.environ.get("CHARSHEET_EDIT_SIZE", str(_DEFAULT_EDIT_SIZE)))
    except ValueError:
        return _DEFAULT_EDIT_SIZE


def preprocess_image(image: Image.Image) -> Image.Image:
    return qwen_image.preprocess_image(image)


def generate_view(
    image: Image.Image,
    prompt: str,
    seed: int = 0,
    negative_prompt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    progress_extra: Optional[dict] = None,
) -> dict:
    """1方向分の画像を生成する(mode="edit_charsheet"、angles adapter 有効)。

    呼び出し前に ensure_edit_loaded() 済みであること(ジョブ側で8方向まとめて
    1回だけロードする設計。ここでは load() を呼ばない)。
    """
    size = _edit_size() if (width is None and height is None) else None
    req = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 4,
        "cfg": 1.0,
        "seed": seed if seed and seed > 0 else -1,
        "lightning": True,
        "shift": None,
        "width": width if width is not None else size,
        "height": height if height is not None else size,
        "_images": [image],
        "_progress_extra": progress_extra,
    }
    family = registry.get("qwen_image")
    return family.generate({"mode": "edit_charsheet", **req})


def get_edit_pipeline_family_name() -> str:
    return "qwen_image"


def ensure_edit_loaded() -> None:
    """Edit グループをロードする(未ロードなら)。ジョブ開始前に1回だけ呼ぶ。

    registry.load() 経由で呼ぶことで、T2I ⇔ Edit の fuse 衝突対策
    (families/qwen_image/family.py)と joyai との相互排他が効く。
    """
    registry.load("qwen_image", "edit_charsheet")


def acquire_generation_lock(blocking: bool = True) -> bool:
    return gpu.generation_lock.acquire(blocking=blocking)


def release_generation_lock() -> None:
    gpu.generation_lock.release()


def get_load_info() -> dict:
    """旧 pipeline.get_load_info() 互換(Edit グループの状態を返す)。"""
    from families.qwen_image import state

    group = state.edit_group
    return {
        "loaded": group.get("loaded", False),
        "quant": group.get("quant"),
        "fallback": group.get("fallback", False),
        "lightning_merged": group.get("lightning_merged"),
        "load_time_s": group.get("load_time_s"),
        "angles_lora": True,
        "angles_available": group.get("angles_available"),
        "method": "edit-angles-adapter",
    }
