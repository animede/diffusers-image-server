# -*- coding: utf-8 -*-
"""
Edit(QwenImageEditPlusPipeline)を参照画像なし(image=None、テキストのみ生成)で
呼び出すためのランタイムパッチ。

背景(CLAUDE.md 61番参照): QwenImageEditPlusPipeline.__call__() の内部機構
(_get_qwen_prompt_embeds の image なしテンプレート、prepare_latents(None) ->
image_latents=None、denoise ループの `if image_latents is not None` ガード)は
すでに image=None に対応済みだが、__call__ 冒頭の2箇所だけが未対応で例外になる:

  1. `image_size = image[-1].size if isinstance(image, list) else image.size`
     -> image=None だと AttributeError。
  2. `condition_images` / `vae_images` / `vae_image_sizes` / `condition_image_sizes` が
     `if image is not None: ...` ブロックの中でのみ定義される -> NameError。

diffusers 本体(venv)は無改変のまま、`inspect.getsource()` で取得した `__call__` の
ソーステキストを上記2箇所だけ最小パッチして `exec` し直す(CLAUDE.md 46番の
「ランタイムパッチ」先例と同じ方針: diffusers/transformers 本体には一切手を入れず、
呼び出し側だけでソースを差し替える)。

設計上の重要点: このパッチ済み `__call__` は「画像0枚のテキストのみ生成」の
リクエストのときだけ使う。画像ありの通常 Edit は従来どおり `pipe(image=[...], ...)`
をそのまま呼ぶ(このモジュールには一切触れない)。パッチ済み関数は
`state.edit_group["text_only_call"]` にキャッシュし、リクエストのたびに exec しない。
"""
import inspect
import textwrap

from families.qwen_image import state

_PATCH1_OLD = (
    "        image_size = image[-1].size if isinstance(image, list) else image.size\n"
)
_PATCH1_NEW = (
    "        if image is None:\n"
    "            image_size = (width or 1024, height or 1024)\n"
    "        else:\n"
    "            image_size = image[-1].size if isinstance(image, list) else image.size\n"
)

_PATCH2_OLD = (
    "        if image is not None and not (isinstance(image, torch.Tensor) "
    "and image.size(1) == self.latent_channels):\n"
)
_PATCH2_NEW = (
    "        condition_images = None\n"
    "        vae_images = None\n"
    "        vae_image_sizes = []\n"
    "        condition_image_sizes = []\n" + _PATCH2_OLD
)


def _build_patched_call():
    """QwenImageEditPlusPipeline.__call__ の image=None 対応版を組み立てる。

    パッチポイントが見つからない場合(将来の diffusers 更新でソースが変わった等)は
    明確な RuntimeError を送出する。呼び出し元(get_text_only_edit_call()）はこれを
    テキストのみ生成の要求時にのみ呼ぶため、画像ありの通常 Edit には一切影響しない。
    """
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as ep_mod

    src = inspect.getsource(QwenImageEditPlusPipeline.__call__)

    if _PATCH1_OLD not in src:
        raise RuntimeError(
            "Edit の画像なし(テキストのみ)生成パッチが適用できません"
            "(diffusers の QwenImageEditPlusPipeline.__call__ 内でパッチ対象1が"
            "見つかりませんでした。diffusers が更新された可能性があります。"
            "画像を1枚以上添付して通常の Edit をお試しください)。"
        )
    src = src.replace(_PATCH1_OLD, _PATCH1_NEW)

    if _PATCH2_OLD not in src:
        raise RuntimeError(
            "Edit の画像なし(テキストのみ)生成パッチが適用できません"
            "(diffusers の QwenImageEditPlusPipeline.__call__ 内でパッチ対象2が"
            "見つかりませんでした。diffusers が更新された可能性があります。"
            "画像を1枚以上添付して通常の Edit をお試しください)。"
        )
    src = src.replace(_PATCH2_OLD, _PATCH2_NEW)

    src = textwrap.dedent(src)
    ns = dict(vars(ep_mod))
    exec(compile(src, "<qwen_image_edit_plus_text_only_call>", "exec"), ns)  # noqa: S102
    return ns["__call__"]


def get_text_only_edit_call():
    """パッチ済み __call__ を edit_group にキャッシュして返す(初回のみ exec)。"""
    edit_group = state.edit_group
    cached = edit_group.get("text_only_call")
    if cached is not None:
        return cached
    patched = _build_patched_call()
    edit_group["text_only_call"] = patched
    return patched
