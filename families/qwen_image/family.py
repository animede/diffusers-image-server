# -*- coding: utf-8 -*-
"""
QwenImageFamily: core.registry.ModelFamily の実装(T2I / I2I / Edit / ControlNet Inpaint のみ)。

diffusers-server(親リポジトリ)からの trim 版: Layered・旧3変種charsheet方式
(edit_angles/edit_angles_bf16/edit_angles_bf16group)は削除した。charsheet は
mode="edit_charsheet" として、edit グループのロードをそのまま流用しつつ
(ロードは通常Editと同一の edit_mod.get_edit_pipeline())、generate() 時に
Multiple-angles LoRA アダプタを追加で有効化する
(families/qwen_image/generate.py の run_edit_charsheet()、
families/qwen_image/edit.py の set_edit_angles() 参照)。

mode に対応するロード関数(core.registry.FamilyRegistry.load() 経由で呼ばれる):
  "t2i" / "i2i"     -> t2i.get_t2i_pipeline() / get_i2i_pipeline()
  "edit"            -> edit.get_edit_pipeline()
  "edit_charsheet"  -> edit.get_edit_pipeline()(edit と同一グループ。ロードは共通)
  "inpaint"         -> controlnet.get_controlnet_inpaint_pipeline()(Inpaintingのみ提供、
                       Union/canny は本サーバでは提供しない)

generate(request) の request は {"mode": <str>, ...パラメータ, "_images": [...]} 形式
(app.py 側であらかじめ画像を読み込み・前処理してから渡す)。
"""
from core.registry import ModelFamily

from families.qwen_image import controlnet as controlnet_mod
from families.qwen_image import edit as edit_mod
from families.qwen_image import generate as generate_mod
from families.qwen_image import lifecycle
from families.qwen_image import state
from families.qwen_image import t2i as t2i_mod
from families.qwen_image.paths import is_2512_model
from families.qwen_image.runtime import is_fp8_lightning_quant, quant_suffix


class QwenImageFamily(ModelFamily):
    name = "qwen_image"

    # ------------------------------------------------------------------
    # T2I <-> Edit のグループ間排他解放(親リポジトリ CLAUDE.md 18番・22番・32番・66番の
    # 知見を trim して踏襲): fp8-lightning 系のロードは「bf16 transformer(約38GB)を
    # 直接cuda:0へロード -> LoRA fuse -> fp8 layerwise cast」のため、fuse完了までの
    # 一時ピークだけで38GB超を要する。別グループ(T2I~19GB常駐 + 共有~15.7GB等)が
    # 常駐したままだと合算でOOMする。GGUF(fuseしない)どうしでも24GB級では合算OOMする
    # ため(親リポジトリ66番)、quant_suffix(GGUF判定)真の場合も事前解放の対象に含める。
    # ------------------------------------------------------------------

    @staticmethod
    def _t2i_load_will_fuse() -> bool:
        """これから行う T2I ロードが bf16 fuse(一時ピーク~38GB)を伴うか。
        2512 は常に fuse、無印は DS_QUANT=fp8-lightning のとき fuse
        (t2i.py の _load_t2i_group_locked() の分岐と同じ判定)。"""
        cfg = state.get_runtime_config()
        return is_2512_model(cfg.t2i_model) or is_fp8_lightning_quant(cfg.quant)

    def _unload_edit_group_for_t2i_fuse(self) -> None:
        """T2Iグループのロード(fuse経路)に先立ち、Editグループ+共有を解放する。"""
        if state.t2i_group["loaded"]:
            return  # ロード済みなら fuse は走らない(解放不要)
        if not self._t2i_load_will_fuse():
            return
        if state.edit_group["loaded"]:
            print(
                "[families.qwen_image] auto-unloading edit group before T2I fuse load "
                "(bf16 fuse一時ピーク~38GBと他グループは共存不可)"
            )
            lifecycle.unload("edit")
            lifecycle.unload_shared_if_unused()

    @staticmethod
    def _unload_t2i_group_for_edit_load(edit_already_loaded: bool) -> None:
        """Editグループのロード(fuse/GGUF経路)に先立ち、T2I/ControlNetグループ+共有を
        解放する。edit_already_loaded はこれからロードするEditグループが既にロード
        済みかどうか(ロード済みならロードは走らず解放不要)。
        """
        if edit_already_loaded:
            return
        if state.t2i_group["loaded"] or state.controlnet_union_group["loaded"] or state.controlnet_inpaint_group["loaded"]:
            print(
                "[families.qwen_image] auto-unloading t2i (+controlnet) group before Edit load "
                "(fuse一時ピーク/GGUF合算とT2Iグループ+共有は低VRAM環境で共存不可)"
            )
            lifecycle.unload("t2i")  # ControlNet -> T2I の順で解放される
            lifecycle.unload_shared_if_unused()

    def load(self, mode: str, model: "str | None" = None, quant: "str | None" = None) -> None:
        """model: T2I/I2Iの場合のみ有効("2512"(既定) | "qwen-image")。他モードでは無視する。

        quant: T2I/I2I/edit で有効。UI のモデルセレクタから bf16/fp8-lightning ⇔ GGUF を
        選べるようにする任意パラメータ(未指定/"default" は現状維持)。
        """
        quant = quant if quant else None
        if mode in ("t2i", "i2i"):
            t2i_mod._switch_t2i_model_if_needed(model)
            t2i_mod._switch_t2i_quant_if_needed(quant)
            self._unload_edit_group_for_t2i_fuse()
            if mode == "t2i":
                t2i_mod.get_t2i_pipeline(model, quant)
            else:
                t2i_mod.get_i2i_pipeline(model, quant)
        elif mode in ("edit", "edit_charsheet"):
            edit_mod._switch_edit_quant_if_needed(quant)
            current_quant = state.get_runtime_config().quant
            if is_fp8_lightning_quant(current_quant) or quant_suffix(current_quant):
                self._unload_t2i_group_for_edit_load(state.edit_group["loaded"])
            edit_mod.get_edit_pipeline(quant)
        elif mode == "inpaint":
            # ControlNet Inpainting は無印ベース(t2i.py 参照)。
            t2i_mod._switch_t2i_model_if_needed("qwen-image")
            self._unload_edit_group_for_t2i_fuse()
            controlnet_mod.get_controlnet_inpaint_pipeline()
        else:
            raise ValueError(f"unknown qwen_image mode: {mode!r}")

    def generate(self, request: dict) -> dict:
        mode = request["mode"]
        if mode == "t2i":
            return generate_mod.run_t2i(request)
        if mode == "i2i":
            return generate_mod.run_i2i(request, request["_image"])
        if mode == "edit":
            return generate_mod.run_edit(request, request["_images"])
        if mode == "edit_charsheet":
            return generate_mod.run_edit_charsheet(request, request["_images"])
        if mode == "inpaint":
            return generate_mod.run_inpaint(request, request["_image"], request["_mask_image"])
        raise ValueError(f"unknown qwen_image mode: {mode!r}")

    def unload(self) -> dict:
        return lifecycle.unload("all")

    def unload_target(self, target: str) -> dict:
        """/api/unload の target(t2i/edit/controlnet/all)による部分解放。"""
        return lifecycle.unload(target)

    def status(self) -> dict:
        return lifecycle.get_status()

    def is_loaded(self) -> bool:
        return lifecycle.is_any_loaded()
