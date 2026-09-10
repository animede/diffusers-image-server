# diffusers-image-server

[diffusers-server](/home/animede/diffusers-server)(port 8601)から **Edit 系機能だけを
独立させた軽量画像サーバ**(port 8620)。venv とモデル実体(ComfyUI ディレクトリ +
HF キャッシュ)は元サーバと共有する。

## 起動

```bash
cd /home/animede/diffusers-image-server
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8620
# LLMプロンプト支援を使う場合
DS_LLM_URL=http://127.0.0.1:64650 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8620
# 24GB級カードで GGUF 既定にする場合
CUDA_VISIBLE_DEVICES=1 DS_QUANT=gguf-q4_k_m venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8620
# 20GB級カードで参照画像ありEditを使う場合
CUDA_VISIBLE_DEVICES=1 DS_QUANT=gguf-q4_k_m DS_OFFLOAD=model_cpu venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8620
```

UI: http://localhost:8620/ (タブ: 画像生成/編集(Edit統合)・Inpaint・アウトペイント・キャラシート)

## API 一覧

| エンドポイント | 内容 |
|---|---|
| `POST /api/edit` | Qwen-Image-Edit-2511。images 0枚=テキストのみ生成、1〜3枚=指示編集、`strength` 指定+1枚=Edit-I2I。`quant` でリクエスト単位切替 |
| `POST /api/t2i` | T2I(model=`2512`(既定)/`qwen-image`、`quant` 切替可) |
| `POST /api/i2i` | I2I(同上) |
| `POST /api/joyai/edit` | JoyAI-Image-Edit-Plus(マルチ参照 1〜3枚、`quant` 切替可) |
| `POST /api/inpaint` | ControlNet Inpainting(image + mask_image、白=再生成領域) |
| `POST /api/outpaint` | 画角拡張(Inpaint 流用。実行前に全モデル解放するため毎回 ~50s のロードコスト) |
| `POST /api/remove_bg` | 背景除去(rembg / anime、CPU) |
| `POST /api/charsheet/generate` ほか `/api/charsheet/*` | キャラシート8方向(ジョブ式。refine/undo/remove_bg/split/sheet/zip) |
| `POST /api/prompt/enhance` `/api/prompt/translate` | LLMプロンプト支援(`DS_LLM_URL`) |
| `GET /api/status` `GET /api/progress` `POST /api/unload` | 運用系(unload target: t2i/edit/controlnet/joyai/all) |

`quant` の値: 未指定/`default`=現状維持、`gguf-q4_k_m` 等、`fp8-lightning`、`none`(bf16)。
fp8-lightning への切替は空きVRAM < 44GB なら 400 を返す(GGUF を案内)。

## キャラシート(新方式)

元サーバの edit_angles 系3方式は持たず、**Edit グループ + fal 2511 Multiple-Angles
LoRA の transformer-level アダプタ**で8方向を生成する(通常 Edit とモデル共有、
切替はアダプタトグルのみで再ロード無し)。プロンプトは構造化タグ
`<sks> [azimuth] eye-level shot medium shot` 8種。

## VRAM 目安(実測)

| 構成 | 常駐/ピーク |
|---|---|
| Edit fp8-lightning(既定、48GB級以上) | ピーク ~35GB(1024²、TE退避込み) |
| Edit GGUF Q4_K_M full tier(空き>=30GB) | ピーク ~30GB |
| Edit GGUF Q4_K_M gguf_te_fp8 tier(24GB級、TE nf4) | ピーク ~20.6GB |
| Edit GGUF Q4_K_M model_cpu tier(20GB級) | ピーク 17.30GB(参照画像3枚、512²、4 steps) |
| charsheet(Edit GGUF + lightning+angles アダプタ、96GB機) | ピーク ~38.8GB(probe実測、bf16 TE 常駐) |
| JoyAI bf16 | ピーク ~33GB |
| JoyAI GGUF Q4_K_M | ピーク ~20.8GB |
| T2I 2512 fp8-lightning | ピーク ~35GB |
| Inpaint / アウトペイント(tiled VAE 既定) | ピーク ~39GB(1280×720) |

### 20GB級カードでのQ4_K_M model_cpu運用

RTX 4000 SFF Ada 20GBで参照画像ありEditを使う場合は、`Q4_K_M` transformerと
`DS_OFFLOAD=model_cpu`を使用する。Q3_K_M全常駐では参照画像1枚・768²のみ成功したが、
896²および参照画像3枚・512²でOOMしたため、汎用運用には採用しない。model_cpu tierは
transformerと共有コンポーネントを使用段階に応じてCPU/GPU間で移動するため低速になるが、
全常駐用に量子化精度をQ3へ下げる必要がなく、Q4_K_Mの品質を維持できる。
参照画像3枚・512²・4 stepsは実機で生成に成功(77.90s、ピーク17.30GB)。

20GB級の空きVRAMは自動tier判定の既定閾値21.5GBを下回るため、transformerをGPU常駐させる
には`DS_OFFLOAD=gguf_te_fp8`を明示すること。Inpaint / アウトペイントはQ3_K_L Edit経路を
使用せず、実測ピーク約39GBのControlNet構成をロードするため、20GBカードでは利用不可。

## 注意

- コア(core/ families/ apps/charsheet のジョブ機構)は diffusers-server からのコピー。
  **コア修正は両リポジトリへ反映すること**(CLAUDE.md「二重管理注意」参照)。
- venv へ `pip install` しないこと(共有 venv)。
- 詳細な運用知見は本リポジトリ CLAUDE.md と元リポジトリ CLAUDE.md(1〜69番)を参照。

## ライセンス

このリポジトリで独自に実装したアプリケーションコードは [Apache License 2.0](LICENSE) で提供します。

> [!IMPORTANT]
> Apache License 2.0 は Qwen-Image / Qwen-Image-Edit 系モデルの重み、派生 LoRA・チェックポイント、その他の第三者製コンポーネントには適用されません。各モデルのライセンス(Qwen-Image 系は Apache 2.0、fal の Multiple-Angles LoRA は Apache 2.0 等)を個別に確認してください。
