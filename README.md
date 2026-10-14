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
| `POST /api/iris/t2i` | Iris-3B Turbo T2I(JSON: prompt / width / height / seed / auto_size。4 steps 固定、negative・cfg なし) |
| `POST /api/klein/generate` | FLUX.2 klein 4B/9B(multipart: prompt / model=4b\|9b / images 0〜8枚 / width / height / seed。4 steps 固定、negative・cfg なし)。参照0枚で T2I、1〜8枚で I2I / 複数参照 |
| `POST /generate/` | Momo Song v5 互換(multipart form → PNG 本体)。Iris-3B で、要求の縦横比のまま約1MPで生成 |
| `POST /api/inpaint` | ControlNet Inpainting(image + mask_image、白=再生成領域) |
| `POST /api/outpaint` | 画角拡張(Inpaint 流用。実行前に全モデル解放するため毎回 ~50s のロードコスト) |
| `POST /api/remove_bg` | 背景除去(rembg / anime、CPU) |
| `POST /api/charsheet/generate` ほか `/api/charsheet/*` | キャラシート多方向(ジョブ式。`views` CSV省略時8方向、`size`省略時1024、256〜2048pxで指定可。refine/undo/remove_bg/split/sheet/zip) |
| `POST /api/prompt/enhance` `/api/prompt/translate` | LLMプロンプト支援(`DS_LLM_URL`) |
| `GET /api/status` `GET /api/progress` `POST /api/unload` | 運用系(unload target: t2i/edit/controlnet/joyai/iris/klein/all) |

`quant` の値: 未指定/`default`=現状維持、`gguf-q4_k_m` 等、`fp8-lightning`、`none`(bf16)。
fp8-lightning への切替は空きVRAM < 44GB なら 400 を返す(GGUF を案内)。

## キャラシート(新方式)

元サーバの edit_angles 系3方式は持たず、**Edit グループ + fal 2511 Multiple-Angles
LoRA の transformer-level アダプタ**で8方向を生成する(通常 Edit とモデル共有、
切替はアダプタトグルのみで再ロード無し)。プロンプトは構造化タグ
`<sks> [azimuth] eye-level shot medium shot` 8種。`size<=768`の小画像生成時は、
全身の切り取りを防ぐためLoRA既定語彙の`wide shot`へ自動的に切り替える。

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
| Iris-3B Turbo int8(既定) | 常駐 ~7.2GB / ピーク ~7.8GB(1344×768、約1.0s/枚、RTX PRO 5000) |
| FLUX.2 klein 4B(GGUF Q4_K_M + TE NF4、全常駐) | ピーク ~7.5GB(1024²、T2I 1.4〜1.6s / 参照2枚 2.3s、RTX PRO 5000) |
| FLUX.2 klein 9B(GGUF Q3_K_M + TE NF4、全常駐) | ピーク ~12.7GB(1024²、T2I 3.1s / 参照1枚 3.8s) |

## Iris-3B Turbo(T2I)

[Iris-3B](https://huggingface.co/speridlabs/iris-3b)(VAEなしの画素空間DiT 3B、テキストエンコーダ
Qwen3-VL-4B-Instruct)に [4-step Turbo LoRA](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora)
をロード時にマージして生成する。diffusers のパイプラインではないため、推論コード
[iris-3b-turbo](https://github.com/pentaCoxian/iris-3b-turbo) のチェックアウトを `sys.path` に足して使う
(共有 venv へは pip install しない。本 venv の torch 2.9 / transformers 5.1 で動作確認済み)。

```bash
git clone https://github.com/pentaCoxian/iris-3b-turbo ~/iris-3b-turbo
```

| 環境変数 | 既定値 | 意味 |
|---|---|---|
| `DS_IRIS_CODE_DIR` | `~/iris-3b-turbo` | 推論コードのチェックアウト |
| `DS_IRIS_CHECKPOINT` | `speridlabs/iris-3b` | HF ID(T2I本体の `model.safetensors` と `config.yaml` だけ取得)またはディレクトリ |
| `DS_IRIS_LORA` | `pentacoxian-dev/iris-3b-turbo-lora` | Turbo LoRA。`owner/repo[:name]` またはローカル `.safetensors` |
| `DS_IRIS_QUANT` | `int8` | `int8`(DiT幹+テキストエンコーダを重みint8、常駐 ~7.2GB)/ `none`(bf16、~13GB) |

- **Qwen 系・JoyAI とは同時常駐しない**(排他グループ)。切替時は registry が相手を自動 unload する
  (Iris の再ロードは約10秒)。
- UI: 「画像生成/編集」タブのモデル選択で「Iris-3B Turbo」。参照画像・ネガティブプロンプト・steps・cfg は
  使わない。英語の説明文プロンプトが最もよく効く。
- **Momo Song v5 から使う**: Momo Song の `.env` を `IMAGE_BACKEND=remote`、
  `IMAGE_API_URL=http://127.0.0.1:8620`、`IMAGE_ENGINE=iris` にすると、背景画像を本サーバの
  `POST /generate/` で生成する(896×512 の要求は 1344×768 で返る)。GPU 使用中は他 API と同じく 409 を返し、
  Momo Song 側は選択中の初期背景を維持する。

## FLUX.2 klein 4B / 9B(T2I・I2I・複数参照)

作者のローカルアプリ Klein Reference Studio(非公開、`~/flux2-klein-studio`)の
推論を本サーバのファミリーとして統合したもの。モデル構成・固定リビジョン・参照画像の自動縮小規則は
移植元と同じ。

| 選択値 | transformer | テキストエンコーダ | ライセンス |
|---|---|---|---|
| `4b`(既定) | GGUF Q4_K_M | Qwen3-4B NF4 | Apache-2.0 |
| `9b` | GGUF Q3_K_M | Qwen3-8B NF4 | **FLUX Non-Commercial License(非商用)** |

| 環境変数 | 既定値 | 意味 |
|---|---|---|
| `DS_KLEIN_MODELS_DIR` | `~/flux2-klein-studio/models` | `klein-4b` / `klein-9b`(diffusers 形式)と `gguf/*.gguf` の置き場。無ければ固定リビジョンを HF キャッシュへ取得 |
| `DS_KLEIN_OFFLOAD` | `none` | `none`(全常駐、高速)/ `model_cpu`(CPU オフロード、低速・省VRAM) |

- 4 steps・guidance 1.0 固定。参照画像は 4B で最大辺 768px(4 枚超は縮小)、9B は画素予算が半分。
- **移植元の `torch.cuda.set_per_process_memory_fraction()`(7GiB 上限)は使わない**。プロセス全体に効き、
  同居する Qwen 系を壊すため。省 VRAM が要る環境では `DS_KLEIN_OFFLOAD=model_cpu`。
- Qwen 系・JoyAI・Iris とは同時常駐しない(排他グループ)。4B と 9B もファミリー内で入れ替える
  (切替は解放 → 再ロードで約20秒)。
- UI: 「画像生成/編集」タブのモデル選択で「FLUX.2 klein 4B / 9B」。9B 選択時は非商用の注意を表示する。

### 画質評価(2026-10-14)

本サーバの他モデルと、同一プロンプト・同一 seed(42)・1024×1024 で比較した(各 1 枚。seed による揺れがあるため傾向として扱うこと)。
環境: RTX PRO 5000 Blackwell 48GB(GPU1)、klein は全常駐。

テキストのみ(T2I)6 種:

| | klein 4B | klein 9B | Iris-3B Turbo | Qwen-Image-2512(Lightning 4 steps) |
|---|---|---|---|---|
| 生成時間 / ピーク VRAM | 1.4s / 7.5GB | 3.1s / 12.7GB | 1.0s / 7.8GB | 8.7s / 34.8GB |
| 実写人物(老人の肖像) | ◎ 肌の質感が最も細かい | ◎ | ○ | ○ 年齢がやや若く出た |
| アニメ | ◎ | ◎ | ○ | ◎ 構図が最も整う |
| 文字「OPEN 24 HOURS」 | ✕「HORS」と綴りを誤った | ◎ | ◎ | ◎ |
| 位置関係(立方体・球・ピラミッド) | ◎ | ◎ | ◎ | ◎ |
| 手(ピアノ) | ○ 指の重なりがやや曖昧 | ◎ | ✕ 3 本目の手 | ○ |
| 風景 | ◎ 光と霧の雰囲気が最良 | ◎ | △ ボートが 2 艘に増えた | ◎ |

参照画像あり(キャラ参照 1 枚で場面変更 / キャラ+背景の 2 枚合成):

| | klein 4B | klein 9B | Qwen-Image-Edit-2511 |
|---|---|---|---|
| 生成時間(1 枚 / 2 枚) | 1.9s / 2.6s | 3.9s / 5.2s | 12.5s / 15.9s |
| 場面変更 | ◎ 同一性を保持 | ◎ | ◎ 参照元の装飾まで保持 |
| 2 枚合成 | 背景をほぼそのまま保ちキャラを小さく配置(手足がやや甘い) | 同方針で 4B よりきれい | 背景を描き直しキャラを大きく配置(参照元の照明が混ざる) |

使い分けの目安:
- **klein 4B**: 速度・VRAM が最も軽く、実写・アニメ・風景は高品質。文字の綴りと細かい手が弱点なので、文字入れ用途には向かない。
- **klein 9B**: 今回の範囲では 4B の弱点(文字・手)が出ず、最もバランスが良い。ただし非商用ライセンス(下記)。
- 合成は「用意した背景を壊さずに置く」なら klein、「キャラを大きく見せ背景は作り直してよい」なら Qwen-Image-Edit。

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

本リポジトリのソースコードの著作権者: © 2026 YUZUKI (animede)。著作権表示と [NOTICE](NOTICE) の内容は、Apache-2.0 第4条に基づき再配布時も保持してください。

このリポジトリで独自に実装したアプリケーションコードは [Apache License 2.0](LICENSE) で提供します。

> [!IMPORTANT]
> Apache License 2.0 は本リポジトリのアプリケーションコードにだけ適用されます。本リポジトリはモデルの重みを含まず、
> 実行時に各配布元から取得したモデルを使います。モデルの重み・LoRA・量子化ファイルはそれぞれの配布元のライセンスに
> 従ってください。

### 使用するモデルのライセンス(2026-10-14 時点、Hugging Face の各モデルカードで確認)

| モデル(配布元) | ライセンス |
|---|---|
| Qwen-Image / Qwen-Image-2512 / Qwen-Image-Edit-2511(Qwen)、ComfyUI 版(Comfy-Org)、GGUF 版(unsloth) | Apache-2.0 |
| Lightning LoRA(lightx2v: Qwen-Image / 2512 / Edit-2511) | Apache-2.0 |
| Multiple-Angles LoRA(fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) | Apache-2.0 |
| ControlNet Inpainting(InstantX/Qwen-Image-ControlNet-Inpainting) | Apache-2.0 |
| JoyAI-Image-Edit-Plus(jdopensource)、GGUF 版(vantagewithai) | Apache-2.0 |
| Iris-3B(speridlabs)、Iris-3B Turbo LoRA(pentacoxian-dev)、Qwen3-VL-4B-Instruct(Qwen) | Apache-2.0 |
| FLUX.2 klein 4B(black-forest-labs)、GGUF 版(unsloth/FLUX.2-klein-4B-GGUF) | Apache-2.0 |
| **FLUX.2 klein 9B**(black-forest-labs、ミラー unsloth/FLUX.2-klein-9B)、GGUF 版(unsloth/FLUX.2-klein-9B-GGUF) | **FLUX Non-Commercial License v2.1** |
| 背景除去 anime-seg(skytnt/anime-seg) | Apache-2.0 |

#### FLUX.2 klein 9B(FLUX Non-Commercial License)の主な条件

ライセンス本文は配布元のモデルリポジトリに同梱されています(要約であり、正確な条件は本文を確認してください)。

- **モデルの利用は非商用目的に限られます**(個人の研究・実験・趣味、営利企業による評価・テストなど。
  直接・間接に対価を得る利用は不可)。商用利用には Black Forest Labs の別ライセンスが必要です(https://bfl.ai/licensing)。
- **生成物(Outputs)は派生物に含まれず、商用目的を含めて利用できます**。ただし FLUX と競合するモデルの
  学習・ファインチューニング・蒸留には使えません。
- 生成物を作成・利用する条件として、違法・権利侵害コンテンツの生成や拡散を防ぐ**コンテンツフィルタを設けるか、
  公開・配布の前に生成物を確認すること**、および法令で求められる場合は AI 生成である旨を表示することが求められます。
  本サーバはコンテンツフィルタを内蔵していないため、9B の生成物は利用者が確認してから使ってください。
- モデルや派生物を配布する場合は、次の帰属表示(Attribution Notice)を目立つ形で添える必要があります:

  > This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Non-Commercial License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

商用で FLUX.2 klein を使う場合は、Apache-2.0 の **4B** を選んでください(API の既定も 4B)。
