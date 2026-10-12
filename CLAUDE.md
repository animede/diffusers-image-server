# CLAUDE.md — diffusers-image-server

diffusers-server(`/home/animede/diffusers-server`、port 8601)から **Edit 系機能だけを
独立させた軽量サーバ**(port **8620**)。元サーバとは venv・モデル実体を共有し、並存する。

**詳細な歴史・実測・トラブルシュートの知見は元リポジトリの
`/home/animede/diffusers-server/CLAUDE.md`(1〜69番)を必ず参照すること。**
本ファイルはこのサーバに関係する知見の凝縮版のみを持つ。

## 【最重要】二重管理注意

`core/` `families/qwen_image/` `families/joyai/` `apps/charsheet/`(ジョブ機構)・
`apps/outpaint.py` は diffusers-server からのコピーが土台。**これらのコア部分に
バグ修正・改善を入れた場合は、両リポジトリ(diffusers-server と本リポジトリ)へ
反映すること**(import 共有ではなくコピー方式のため、片方だけ直すと乖離する)。
逆に diffusers-server 側で qwen_image/joyai のコア修正が入った場合も本リポジトリへ
反映が必要。ただし本リポジトリ独自の変更(下記「本リポジトリ独自の設計」)は
diffusers-server 側には存在しないので混ぜないこと。

## 起動・環境

- 起動: `venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8620`
- `venv` は `/home/animede/diffusers-server/venv` へのシンボリックリンク
  (diffusers 0.40.0.dev0 git版 + comfy-env 継承)。**comfy-env / venv に直接
  `pip install` しないこと**(元 CLAUDE.md 5番・6番)。
- モデル実体は共有: `DS_COMFYUI_DIR`(既定 `~/ComfyUI`)の
  `models/diffusion_models/`・`models/loras/` を優先参照、無ければ HF Hub から自動DL
  (`core/resolve.py`)。モデルの複製はしない。
- LLM プロンプト支援は `DS_LLM_URL`(既定 `http://127.0.0.1:64652`)。
  本番運用では `DS_LLM_URL=http://127.0.0.1:64650` を指定して起動している。

## 機能スコープ

含む: Edit-2511(指示編集/テキストのみ生成/strength I2I)、T2I/I2I(2512・無印)、
JoyAI-Edit-Plus(マルチ参照)、Iris-3B Turbo T2I(+ Momo Song 互換 `/generate/`)、
ControlNet Inpainting、アウトペイント、
キャラクターシート(新方式、下記)、LLMプロンプト支援、背景除去。
いずれもリクエスト単位 quant 切替(bf16/fp8-lightning ⇔ GGUF、元 CLAUDE.md 69番)+
fp8-lightning への切替時 VRAM ガード(空き < 44GB なら 400)対応。

含まない: FLUX.2 / Z-Image / LTX-2.3 / Layered / edit_angles 系3方式 /
scene_angles / tpose / Mage-Flow / ControlNet(Inpaint 以外の Union 系 API)。

## 本リポジトリ独自の設計: キャラシート新方式(Edit + angles adapter)

diffusers-server 本体の `DS_CHARSHEET_METHOD` 4方式(bf16-group 等)は移植していない。
代わりに **Edit グループのロード時に fal 2511 Multiple-Angles LoRA(295MB)を
2本目の transformer-level アダプタとして常時ロードし、リクエスト単位でトグルする**
(`families/qwen_image/edit.py` の `_apply_angles_adapter()` / `set_edit_angles()`、
`generate.py` の `run_edit_charsheet()`、mode="edit_charsheet")。
元リポジトリ scratchpad の probe_gguf_charsheet.py で品質実証済み
(GGUF Q4_K_M で 7.2s/方向、品質は bf16-group と同等判定)。

守るべきロード順序・トグル規約(元 CLAUDE.md 17/59/62/63番の罠):

1. **GGUF 経路**: lightning・angles とも `transformer.load_lora_adapter()` を
   `.to("cuda")` / `enable_model_cpu_offload()` より**前**(CPU上)で呼ぶ。
2. **fp8-lightning 経路**: Lightning fuse(`fuse_lora_into_transformer()`)→
   angles を adapter ロード → `enable_layerwise_casting(fp8_e4m3fn)` の順。
   **fp8 キャスト後に LoRA をロードすると lora_A/lora_B が fp8 dtype で作られ
   `addmm_cuda not implemented for Float8_e4m3fn` で落ちる**。
3. **fal 2511 LoRA は raw の state_dict をそのまま `load_lora_adapter()` に渡すこと**
   (既に "transformer." プレフィックス付き PEFT 形式のため、
   `_convert_non_diffusers_qwen_lora_to_diffusers()` を通すとプレフィックスが二重になり
   `Target modules {...} not found` で失敗する。実機で確認済み、初期実装のバグ)。
4. **トグルは必ず `enable_lora()` → `set_adapters([...])` の順**
   (`set_adapters()` は PEFT の `_disable_adapters` フラグをクリアしないため、
   `enable_lora()` を先に呼ばないと完全 no-op になる。元 62番の実バグ)。
5. **angles 残留リセット**: fp8-lightning 構成では `set_edit_lightning()` が
   アダプタ状態を触らない(fuse 済みで常時 True)ため、通常 Edit / Edit-I2I は
   必ず `set_edit_angles(pipe, False)` を呼んで charsheet の angles を外す
   (`generate.py` の run_edit()/run_edit_i2i() 冒頭)。
6. プロンプトは構造化タグ `<sks> [azimuth] eye-level shot medium shot`
   (`apps/charsheet/prompts.py` VIEWS_LORA_2511、元 67番)。

## 本リポジトリ独自の設計: Iris-3B ファミリー

`families/iris3b/` は diffusers-server に存在しない独自ファミリー(二重管理の対象外)。

- diffusers 非対応のため推論コード iris3b(pentaCoxian/iris-3b-turbo)を `DS_IRIS_CODE_DIR/src` から
  import する。**venv へ pip install しない**(推論に要る torch / transformers / omegaconf は comfy-env にある)。
- `registry.register(family, exclusive_with=["qwen_image", "joyai"])`。48GB の GPU1 では Qwen Inpaint
  ~39GB と Iris ~7.8GB を同居させない。
- `generate_turbo` はステップ callback を持たないため、進捗は `start_generating(mode, 0)`(不確定表示)。
- `POST /generate/` は Momo Song v5 の remote 画像プロトコル(multipart → PNG 本体)。`auto_size=True` で
  要求の縦横比のまま約1MPにする(Turbo の蒸留解像度)。

## 元リポジトリから引き継ぐ主要な罠(要点のみ、詳細は元 CLAUDE.md の番号)

- **registry.load() 経由でないと排他 unload が効かない**(1番)。新しい呼び出し経路は
  必ず `registry.load("qwen_image", mode)` → `family.generate()` の順。
- **Edit は QwenImageEditPlusPipeline に必ずリストで画像を渡す**(2番)。
- **TE CPU退避の GPU 復帰は `pipe(...)` 直前**(23番の `_execution_device` の罠)。
  発振バグ対策の正規化判定(63番)も core/optimize.py に含まれる。
- **state.lock は非再入**(14番)。`_load_*_locked()` は呼び出し側がロック保持前提。
- **unload の解放順序**: ControlNet → T2I(18番)。Edit の `i2i_pipe` も transformer
  参照共有のため一緒に None にする(64番)。
- **T2I ⇔ Edit の fuse 衝突**: fp8-lightning ロードは bf16 ~38GB の一時ピークを持つ。
  切替時は相手グループ+共有コンポーネントを事前解放(22/32/66番、family.py)。
- **transformer を丸ごとホストRAMへ退避する実装は禁止**(33番の事故)。
- **GGUF Edit の3段階 tier**(59番): full(空き>=30GB)/ gguf_te_fp8(>=21.5GB、
  TE nf4)/ model_cpu。24GB 実カードは gguf_te_fp8(nf4)で peak ~20.6GB。
- **processor は Qwen2VLProcessor を明示**(10番)。
- **`pipe.set_adapters([])` の空リストは KeyError**(13番)→ `disable_lora()` を使う。
- **outpaint は実行前に全ファミリー解放**(56番タスク3、app.py。毎回 ~50s の
  再ロードコストを許容する保守的設計)。

## 構成

```
diffusers-image-server/
├── app.py                # FastAPI(port 8620。/api/edit /api/t2i /api/i2i
│                         #   /api/joyai/edit /api/iris/t2i /generate/ /api/inpaint /api/outpaint /api/remove_bg
│                         #   /api/charsheet/* /api/status /api/unload /api/progress /api/prompt/*)
├── core/                 # diffusers-server からコピー(無変更)
├── families/
│   ├── qwen_image/       # 刈り込み版(layered/edit_angles 系削除、angles adapter 追加)
│   ├── joyai/            # コピー(無変更)
│   └── iris3b/           # 本リポジトリ独自(diffusers-server には無い。二重管理対象外)
├── apps/
│   ├── charsheet/        # ジョブ機構はコピー、generate.py のみ新方式に差し替え
│   └── outpaint.py       # コピー(無変更)
├── static/               # 4タブ(画像生成/編集・Inpaint・アウトペイント・キャラシート)
└── outputs/              # 生成物(.gitignore)
```
