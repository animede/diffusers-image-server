# -*- coding: utf-8 -*-
"""
JoyAI-Image-Edit-Plus ファミリーのランタイム構成(環境変数)。

抽出元: tools/joyai_regress.py(ステップ①の回帰確認スクリプト)。families/z_image・
families/ltx2 と同じ「runtime.py に環境変数を集約する」規約に合わせて新設した。

DS_JOYAI_TE_OFFLOAD(既定 "auto"): Qwen3-VL text_encoder(bf16 ~17.5GB)を
プロンプトエンコード後に CPU へ退避するかどうか。
  - "auto"(既定): 48GB専有を想定し常に有効にする(常駐すると transformer(~32.5GB)+
    text_encoder(~17.5GB)で ~50GB となり48GB専有では収まらないため、"auto" は
    実質「常に on」。96GB機など大容量VRAM環境でも退避コスト自体は小さい
    (エンコード時だけの短い窓、CLAUDE.md 39番のIA2V事故のような「丸ごと片道スワップ」
    ではなく毎回のGPU復帰を伴う往復のため、常時オンでも安全側)。
  - "on": 常に有効。
  - "off": 無効(常駐のまま。96GB機で最速を狙う場合のみ推奨)。

DS_JOYAI_QUANT(既定 "none"、2026-08-18追加、CLAUDE.md 68番): transformer の GGUF
量子化。"none"(既定) は現行の bf16 完全常駐(diff レベルで無変更)。
"gguf-q4_k_m" 等(qwen_image の quant_suffix() と同じ語彙、"gguf-" prefix を剥がして
大文字化しGGUFファイル名サフィックスにする)を指定すると、
`vantagewithai/JoyAI-Image-Edit-Plus-ComfyUI-GGUF` の対応する量子化ファイルを
(ComfyUI ディレクトリ優先 + HF フォールバックで)ロードする。

量子化ファイル選択は明示指定のみ(qwen_image Edit の3段階自動判定(59番)のような
空きVRAM閾値ベースの自動tieringは行わない、本タスクではQ4_K_M固定運用を想定)。
"""
from core.config import env_str

DEFAULT_TE_OFFLOAD = "auto"
DEFAULT_QUANT = "none"


class JoyAIRuntimeConfig:
    """環境変数から読み取るランタイム構成。

    DS_JOYAI_TE_OFFLOAD : "auto"(既定) | "on" | "off"
    DS_JOYAI_QUANT       : "none"(既定、bf16) | "gguf-q4_k_m" 等
    DS_DEVICE           : "cuda"(既定)
    """

    def __init__(self):
        self.te_offload = env_str("DS_JOYAI_TE_OFFLOAD", DEFAULT_TE_OFFLOAD).strip().lower()
        self.quant = env_str("DS_JOYAI_QUANT", DEFAULT_QUANT).strip().lower()
        self.device = env_str("DS_DEVICE", "cuda")

    def __repr__(self):
        return (
            f"JoyAIRuntimeConfig(te_offload={self.te_offload!r}, quant={self.quant!r}, "
            f"device={self.device!r})"
        )


def quant_suffix(quant: str) -> "str | None":
    """DS_JOYAI_QUANT の値からGGUFファイル名サフィックス(例: 'gguf-q4_k_m' -> 'Q4_K_M')を
    得る。'none'/空 なら None(bf16経路)を返す(families.qwen_image.runtime.quant_suffix()
    と同じロジック)。
    """
    quant = (quant or "none").strip().lower()
    if quant in ("", "none", "off", "bf16"):
        return None
    if quant.startswith("gguf-"):
        quant = quant[len("gguf-"):]
    elif quant.startswith("gguf_"):
        quant = quant[len("gguf_"):]
    return quant.upper()


_QUANT_KEEP_VALUES = {"", "default", "keep", "current"}


def normalize_quant_request(quant: "str | None") -> "str | None":
    """リクエストの quant パラメータを正規化する(CLAUDE.md 69番、
    families.qwen_image.runtime.normalize_quant_request() と同じ思想)。

    None または "default"/"keep"/"current"/空文字 の場合は None を返す
    (=切替要求なし、現在の JoyAIRuntimeConfig.quant のまま)。それ以外
    ("none" = bf16 を含む)は strip().lower() した文字列をそのまま返す
    ("gguf-q4_k_m" 等)。"none" を明示指定すれば、現在GGUFがロード済みでも
    bf16 へ切り替わる(current との比較で不一致なら通常どおり切替が起きる)。
    """
    if quant is None:
        return None
    value = quant.strip().lower()
    if value in _QUANT_KEEP_VALUES:
        return None
    return value


def should_offload_te(config: "JoyAIRuntimeConfig | None" = None) -> bool:
    """DS_JOYAI_TE_OFFLOAD から、この生成で text_encoder CPU退避を使うか判定する。

    48GB専有では常駐(transformer ~32.5GB + text_encoder ~17.5GB = ~50GB)がそもそも
    収まらないため、"auto" は常に True を返す(qwen_image の should_offload_edit_text_encoder()
    と異なり解像度依存の分岐は設けない。JoyAIは常にtransformerが大きい16Bモデルのため)。
    """
    config = config or JoyAIRuntimeConfig()
    if config.te_offload == "off":
        return False
    return True  # "auto" / "on" とも常に有効
