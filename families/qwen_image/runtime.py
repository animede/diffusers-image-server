# -*- coding: utf-8 -*-
"""
Qwen-Image 系ファミリーのランタイム構成(環境変数)。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - RuntimeConfig とその環境変数名一覧(行291-332)
  - quant_suffix() / is_fp8_lightning_quant()(行335-352)

変更点: 環境変数の読み取りは core.config.env_str/env_bool 経由にした(DS_* 新名を優先し、
旧名 QWENIMG_* にフォールバック)。ロジック・既定値は無変更。

新環境(RTX PRO 5000 48GB 専有)向けの既定値調整(REBUILD_PLAN 申し送り事項):
  Edit/Layered の既定量子化を fp8-lightning にする(REBUILD_PLAN §2.4)。
  T2I の既定は "2512"(2026-07-18変更: 公式 Qwen/Qwen-Image-2512 bf16 + Lightning fuse +
  fp8 layerwise cast。旧既定 "2512-4bit"(コミュニティ製NF4)は品質劣位のため廃止、
  paths.py の T2I_2512_REPO コメント参照)。環境変数で明示指定すれば上書き可能。
"""
from core.config import env_bool, env_str

from families.qwen_image.paths import FP8_LIGHTNING_QUANT_VALUES, LAYERED_DEFAULT_QUANT


# 申し送り事項(REBUILD_PLAN §2.4, タスク1)+ 2026-07-18変更: Edit/Layered = fp8-lightning、
# T2I の既定は "2512"(公式 Qwen/Qwen-Image-2512 bf16 + Lightning fuse + fp8 cast。
# 旧既定 "2512-4bit"(コミュニティ製NF4)は廃止。旧値は後方互換で "2512" として扱う)。
DEFAULT_QUANT = "fp8-lightning"
DEFAULT_T2I_MODEL = "2512"


class RuntimeConfig:
    """環境変数から読み取るランタイム構成。

    DS_OFFLOAD(旧 QWENIMG_OFFLOAD) : "auto"(既定)/ "none" / "group" / "group_lowvram" / "model_cpu"
    DS_ATTN(旧 QWENIMG_ATTN)       : "default"(既定)/ "native" / "_native_flash" / "_native_cudnn" /
                       "_native_efficient" / "flex" / "xformers"
    DS_COMPILE(旧 QWENIMG_COMPILE) : "0"(既定)/ "1"
    DS_DEVICE(旧 QWENIMG_DEVICE)   : "cuda"(既定)
    DS_QUANT(旧 QWENIMG_QUANT)     : "fp8-lightning"(このファミリーの既定。旧実装は"none")/
                       "gguf-q4_k_m" 等 / "none"(bf16のまま)
    DS_T2I_MODEL(旧 QWENIMG_T2I_MODEL) : "2512"(このファミリーの既定。公式2512 fp8-lightning fuse)/
                       "qwen-image"(旧値 "2512-4bit" は後方互換で "2512" 扱い)
    DS_LAYERED_QUANT(旧 QWENIMG_LAYERED_QUANT) : "fp8-lightning"(既定)
    DS_QWEN_TILED_VAE              : "1"(既定、共有VAEのencode/decodeを常時tiled化。
                       CLAUDE.md 56番。outpaint等の大キャンバスVAEエンコードOOM対策) / "0"(旧動作)
    DS_QWEN_TE_QUANT               : "none"(既定、bf16 約16GB)/ "fp8"(共有 text_encoder を
                       fp8_e4m3fn ストレージ + bf16 compute の layerwise casting で約半分にする。
                       16GB級カード対応。LTX-2.3 の DS_LTX2_TE_QUANT と同じ機構)/
                       "nf4"(bnb NF4量子化で常駐 約5.5GBまで縮める。実測ロード時の一時ピークが
                       13.83GiBあるため、GGUF Edit 24GB級ティア(edit.py の GGUF_TIER_TE_FP8)
                       専用に「先にTEをロード→後でtransformer」という逆順ロードで使う設計。
                       CLAUDE.md 59番のnf4化タスク参照)
    """

    def __init__(self):
        self.offload = env_str("DS_OFFLOAD", "auto").strip().lower()
        self.attention_backend = env_str("DS_ATTN", "default").strip().lower()
        self.compile = env_bool("DS_COMPILE", False)
        self.device = env_str("DS_DEVICE", "cuda")
        self.quant = env_str("DS_QUANT", DEFAULT_QUANT).strip().lower()
        self.t2i_model = env_str("DS_T2I_MODEL", DEFAULT_T2I_MODEL).strip().lower()
        self.layered_quant = env_str("DS_LAYERED_QUANT", LAYERED_DEFAULT_QUANT).strip().lower()
        self.tiled_vae = env_bool("DS_QWEN_TILED_VAE", True)
        self.te_quant = env_str("DS_QWEN_TE_QUANT", "none").strip().lower()

    def __repr__(self):
        return (
            f"RuntimeConfig(offload={self.offload!r}, attention_backend={self.attention_backend!r}, "
            f"compile={self.compile}, device={self.device!r}, quant={self.quant!r}, "
            f"t2i_model={self.t2i_model!r}, layered_quant={self.layered_quant!r}, "
            f"tiled_vae={self.tiled_vae}, te_quant={self.te_quant!r})"
        )


def quant_suffix(quant: str) -> "str | None":
    """DS_QUANT の値からGGUFファイル名サフィックス(例: 'gguf-q4_k_m' -> 'Q4_K_M')を得る。
    'none'/空/'bf16'/'fp8-lightning' なら None(GGUF経路ではない)を返す。

    抽出元: pipeline_manager.py quant_suffix()(行335-346)。ロジック無変更。
    """
    quant = (quant or "none").strip().lower()
    if quant in ("", "none", "off", "bf16", "fp8") or quant in FP8_LIGHTNING_QUANT_VALUES:
        return None
    if quant.startswith("gguf-"):
        quant = quant[len("gguf-"):]
    elif quant.startswith("gguf_"):
        quant = quant[len("gguf_"):]
    return quant.upper()


def is_fp8_lightning_quant(quant: str) -> bool:
    """DS_QUANT が fp8-lightning(マージ済み高速化オプション)かどうか。"""
    return (quant or "none").strip().lower() in FP8_LIGHTNING_QUANT_VALUES


# ============================================================================
# リクエスト単位の quant 切替(CLAUDE.md 69番、2026-08-18追加)
# ============================================================================
# UI のモデルセレクタから bf16/fp8-lightning ⇔ GGUF を選べるようにするための正規化
# ヘルパー。24番(T2Iモデル切替)と同じ思想: リクエストの quant パラメータは
# 「未指定/"default"」なら現状維持、明示値なら RuntimeConfig.quant を上書きして
# 該当グループを unload -> 再ロードする。
_QUANT_KEEP_VALUES = {"", "default", "keep", "current"}


def normalize_quant_request(quant: "str | None") -> "str | None":
    """リクエストの quant パラメータを正規化する。

    None または "default"/"keep"/"current"/空文字 の場合は None を返す
    (=切替要求なし、現在の RuntimeConfig.quant のまま)。それ以外は
    strip().lower() した文字列をそのまま返す(RuntimeConfig.quant にそのまま
    代入できる形。"gguf-q4_k_m" 等 quant_suffix() が解釈できる値、または
    "fp8-lightning"/"none" 等)。
    """
    if quant is None:
        return None
    value = quant.strip().lower()
    if value in _QUANT_KEEP_VALUES:
        return None
    return value


def gguf_quant_vram_ok(quant: "str | None") -> bool:
    """fp8-lightning(bf16 transformer ~40GBを直接GPU上でfuseする方式)への切替要求を
    ロード前に弾くための簡易ガード。空きVRAMが core.config.VRAM_FREE_THRESHOLD_GB
    (既定44.0、48GB専有機のfp8-lightning実測ピークに余裕を持たせた下限)未満なら False。
    GGUF/bf16(非fuse)への切替は対象外(常に True)。
    """
    if not is_fp8_lightning_quant(quant):
        return True
    from core import gpu
    from core.config import VRAM_FREE_THRESHOLD_GB

    return gpu.free_vram_gb() >= VRAM_FREE_THRESHOLD_GB
