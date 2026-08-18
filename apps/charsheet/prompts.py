# -*- coding: utf-8 -*-
"""
8方向の視点定義(旧 charsheet/prompts.py 移植)。
各エントリ: key, label_ja, label_en, prompt

Phase 3 での方針転換(ユーザー目視比較の結果に基づく): 当初実装(プロンプトのみ、
Multiple-angles LoRA 廃止)で新旧比較したところ「コスチューム等の一貫性が旧LoRA版
よりかなり劣る」と判定されたため、LoRA 併用(families/qwen_image/edit_angles.py の
fp8-lightning-angles 変種)を復活させた。

2026-08-18(CLAUDE.md 67番): Multiple-angles LoRA を fal/Qwen-Image-Edit-2511-
Multiple-Angles-LoRA(2511ネイティブ)へ差し替えたのに伴い、トリガー文言も
「自然文の視点描写」から fal LoRA が学習した**構造化タグ**
`<sks> [azimuth] [elevation] [distance]` へ書き換えた(事前A/Bで90度回転の成功率が
33%->100%に改善)。distance は "medium shot"(参照ポーズ、全身が写る標準構図)に
統一。elevation は "eye-level shot"(カメラが被写体と同じ高さ)で全方向共通。
DS_ANGLES_LORA=2509 の場合は旧LoRA用の自然文プロンプト(旧 charsheet/prompts.py と
同一文言)を使う(プロンプトとLoRAはペアで切り替わらないと壊れるため、
families.qwen_image.paths.DS_ANGLES_LORA の値を見て分岐する)。

このモジュールは3つの形式を保持する:
  - VIEWS_LORA_2511   : fal 2511 LoRA 用の構造化タグ(既定、DS_ANGLES_LORA=2511)
  - VIEWS_LORA_2509   : 旧 charsheet/prompts.py と同一の自然文(LoRAトリガー文、
                        DS_ANGLES_LORA=2509 の逃げ道用)
  - VIEWS_PROMPT_ONLY : Phase 3 当初実装のプロンプト(全方向末尾に "keep items" 補強
                        文を追加。LoRAなしのプロンプトのみ多アングル生成向け)

VIEWS / VIEW_BY_KEY は apps/charsheet/generate.py の use_angles_lora() と
families.qwen_image.paths.DS_ANGLES_LORA の値に応じてどちらかを指す
モジュールレベル変数(呼び出し側 apps/charsheet/jobs.py が
`from apps.charsheet.prompts import VIEW_BY_KEY, VIEWS` する際に、プロセス起動時の
環境変数の値で選択される)。
"""
from apps.charsheet.generate import use_angles_lora
from families.qwen_image.paths import DS_ANGLES_LORA

NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形"

_KEEP_ITEMS_SUFFIX = (
    " Keep all held items, weapons, and accessories exactly as shown in the "
    "reference image; do not omit or change any props, bags, weapons or "
    "decorations the character is holding or wearing."
)

# --- LoRA使用時・fal 2511ネイティブ(既定): 構造化タグ `<sks> [azimuth] [elevation]
# [distance]`。README(fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA)記載の
# azimuth語彙(front view / front-right quarter view / right side view /
# back-right quarter view / back view / back-left quarter view / left side view /
# front-left quarter view)をそのまま使う。elevation は "eye-level shot" 固定、
# distance は "medium shot" 固定(全身が写る標準構図、README のreference pose)。 ---
VIEWS_LORA_2511 = [
    {
        "key": "front",
        "label_ja": "前",
        "label_en": "Front",
        "prompt": "<sks> front view eye-level shot medium shot",
    },
    {
        "key": "back",
        "label_ja": "後ろ",
        "label_en": "Back",
        "prompt": "<sks> back view eye-level shot medium shot",
    },
    {
        "key": "left",
        "label_ja": "左",
        "label_en": "Left",
        "prompt": "<sks> left side view eye-level shot medium shot",
    },
    {
        "key": "right",
        "label_ja": "右",
        "label_en": "Right",
        "prompt": "<sks> right side view eye-level shot medium shot",
    },
    {
        "key": "front_left_45",
        "label_ja": "左前45度",
        "label_en": "Front-Left 45°",
        "prompt": "<sks> front-left quarter view eye-level shot medium shot",
    },
    {
        "key": "front_right_45",
        "label_ja": "右前45度",
        "label_en": "Front-Right 45°",
        "prompt": "<sks> front-right quarter view eye-level shot medium shot",
    },
    {
        "key": "back_left_45",
        "label_ja": "左後ろ45度",
        "label_en": "Back-Left 45°",
        "prompt": "<sks> back-left quarter view eye-level shot medium shot",
    },
    {
        "key": "back_right_45",
        "label_ja": "右後ろ45度",
        "label_en": "Back-Right 45°",
        "prompt": "<sks> back-right quarter view eye-level shot medium shot",
    },
]

# --- LoRA使用時・旧2509(DS_ANGLES_LORA=2509 の逃げ道): 旧 charsheet/prompts.py と
# 同一文言(自然文のLoRAトリガー文、"keep items" 補強なし)。 ---
VIEWS_LORA_2509 = [
    {
        "key": "front",
        "label_ja": "前",
        "label_en": "Front",
        "prompt": (
            "Show the character in a full body front view, facing directly "
            "toward the camera, neutral standing A-pose with arms slightly "
            "away from the body, full figure visible from head to toe"
        ),
    },
    {
        "key": "back",
        "label_ja": "後ろ",
        "label_en": "Back",
        "prompt": (
            "Show the character from behind in a full body back view, rear "
            "facing the camera, neutral standing pose, showing all back "
            "details, hair, costume and accessories from behind, full "
            "figure head to toe"
        ),
    },
    {
        "key": "left",
        "label_ja": "左",
        "label_en": "Left",
        "prompt": (
            "Show the character in a full body left side profile view, "
            "character facing to the left, neutral standing pose, showing "
            "the complete left side silhouette from head to toe"
        ),
    },
    {
        "key": "right",
        "label_ja": "右",
        "label_en": "Right",
        "prompt": (
            "Show the character in a full body right side profile view, "
            "character facing to the right, neutral standing pose, showing "
            "the complete right side silhouette from head to toe"
        ),
    },
    {
        "key": "front_left_45",
        "label_ja": "左前45度",
        "label_en": "Front-Left 45°",
        "prompt": (
            "Show the character from a 3/4 front-left angle, neutral "
            "standing pose, showing both front and left side details, full "
            "body visible from head to toe"
        ),
    },
    {
        "key": "front_right_45",
        "label_ja": "右前45度",
        "label_en": "Front-Right 45°",
        "prompt": (
            "Show the character from a 3/4 front-right angle, neutral "
            "standing pose, showing both front and right side details, "
            "full body visible from head to toe"
        ),
    },
    {
        "key": "back_left_45",
        "label_ja": "左後ろ45度",
        "label_en": "Back-Left 45°",
        "prompt": (
            "Show the character from a 3/4 back-left angle, neutral "
            "standing pose, showing back and left side details, full body "
            "visible from head to toe"
        ),
    },
    {
        "key": "back_right_45",
        "label_ja": "右後ろ45度",
        "label_en": "Back-Right 45°",
        "prompt": (
            "Show the character from a 3/4 back-right angle, neutral "
            "standing pose, showing back and right side details, full body "
            "visible from head to toe"
        ),
    },
]

VIEWS_LORA = VIEWS_LORA_2509 if DS_ANGLES_LORA == "2509" else VIEWS_LORA_2511

# --- LoRAなし(プロンプトのみ)時: Phase 3 当初実装。小物保持補強つき ---
# 注: プロンプトのみモードはLoRAを使わないため常に自然文(VIEWS_LORA_2509)を土台にする
# (VIEWS_LORA が2511の構造化タグを指していても、prompt-onlyモードでは無関係。
# `<sks> front view eye-level shot medium shot` に "keep items" 補強を付けても
# LoRA無しでは意味を持たないため)。
def _with_keep_items_suffix(prompt: str) -> str:
    base = prompt if prompt.endswith(".") else prompt + "."
    return base + _KEEP_ITEMS_SUFFIX


VIEWS_PROMPT_ONLY = [{**v, "prompt": _with_keep_items_suffix(v["prompt"])} for v in VIEWS_LORA_2509]

VIEWS = VIEWS_PROMPT_ONLY if not use_angles_lora() else VIEWS_LORA
VIEW_BY_KEY = {v["key"]: v for v in VIEWS}

# キャラクターシート合成時の並び順(旧仕様維持)
# 1行目: 前系, 2行目: 後ろ系
SHEET_LAYOUT = [
    ["front", "front_left_45", "front_right_45", "left"],
    ["right", "back_left_45", "back_right_45", "back"],
]
