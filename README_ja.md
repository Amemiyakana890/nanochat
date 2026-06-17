# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat は、LLM を訓練するための最もシンプルな実験用ハーネスです。単一の GPU ノード上で動作するように設計されており、コードは最小限でハック可能、そしてトークン化、事前学習、ファインチューニング、評価、推論、チャット UI といった LLM の主要な段階をすべてカバーしています。例えば、自分自身の GPT-2 相当の能力を持つ LLM（2019 年には訓練に約 43,000 ドルかかりました）をわずか 48 ドル（8XH100 GPU ノードで約 2 時間）で訓練し、見慣れた ChatGPT ライクな Web UI で会話することができます。スポットインスタンスを使えば、総コストは約 15 ドルにまで近づけることができます。より一般的には、nanochat は、たった 1 つの複雑さのダイヤル、すなわち GPT トランスフォーマーモデルのレイヤー数である `--depth` を設定するだけで、計算量最適なモデルのミニシリーズ全体を訓練できるように、すぐ使える状態で構成されています（GPT-2 相当の能力はちょうど depth 26 あたりで実現されます）。その他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、訓練ホライズン、重み減衰、…）は、最適な形で自動的に計算されます。

リポジトリに関する質問については、Devin/Cognition の [DeepWiki](https://deepwiki.com/karpathy/nanochat) を使って質問するか、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)を利用するか、Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルに来てもらうことをおすすめします。

## GPT-2 到達時間リーダーボード

現在、開発の主な焦点は、最も多くの計算量を必要とする事前学習段階のチューニングにあります。modded-nanogpt リポジトリに触発され、進歩とコミュニティの協力を促すために、nanochat は「GPT-2 スピードラン」のリーダーボードを維持しています。これは、DCLM CORE スコアによって測定される GPT-2 グレードの能力に nanochat モデルを訓練するために必要な実時間（ウォールクロック時間）です。[runs/speedrun.sh](runs/speedrun.sh) スクリプトは、GPT-2 グレードのモデルを訓練して会話するための基準となる方法を常に反映しています。現在のリーダーボードは以下の通りです。

| # | 時間 | val_bpb | CORE | 説明 | 日付 | コミット | 貢献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 時間 | - | 0.2565 | オリジナルの OpenAI GPT-2 チェックポイント | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 ベースライン、わずかに過学習 | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 わずかに学習不足 **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 総バッチサイズを 1M トークンに引き上げ | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | データセットを NVIDIA ClimbMix に変更 | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | オートリサーチ [ラウンド 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | オートリサーチ ラウンド 2 | Mar 14 2026 | a825e63 | @karpathy |

私たちが最も重視する指標は「GPT-2 到達時間」、すなわち 8XH100 GPU ノード上で GPT-2 (1.6B) の CORE 指標を上回るために必要な実時間です。GPT-2 の CORE スコアは 0.256525 です。2019 年には GPT-2 の訓練に約 43,000 ドルかかったため、7 年間にわたるスタック全体での多くの進歩のおかげで、今やはるかに高速に、しかも 100 ドルを大きく下回るコストで実現できるというのは驚くべきことです（例えば、現在の約 3 ドル/GPU/時で、8XH100 ノードは約 24 ドル/時なので、2 時間で約 48 ドルです）。

リーダーボードの解釈方法や貢献方法に関するさらなるドキュメントについては、[dev/LEADERBOARD.md](dev/LEADERBOARD.md) を参照してください。

## はじめに

### セットアップ

nanochat は依存関係管理に [uv](https://docs.astral.sh/uv/) を使用します。インストールするには:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

開発用には（pytest、matplotlib、ipykernel、transformers などを追加します）:

```bash
uv sync --extra gpu --group dev
```

### GPT-2 を再現して会話する

最も楽しめるのは、自分自身の GPT-2 を訓練してそれと会話することです。そのための一連のパイプライン全体は、単一のファイル [runs/speedrun.sh](runs/speedrun.sh) に収められており、これは 8XH100 GPU ノード上で実行するように設計されています。お気に入りのプロバイダーから新しい 8XH100 GPU ボックスを起動し（例えば、私は [Lambda](https://lambda.ai/service/gpu-cloud) を使っていて気に入っています）、訓練スクリプトを開始してください:

```bash
bash runs/speedrun.sh
```

これは実行に約 3 時間かかるため、screen セッション内で実行するとよいでしょう。完了したら、ChatGPT ライクな Web UI で会話できます。ローカルの uv 仮想環境が有効になっていることを再度確認し（`source .venv/bin/activate` を実行）、サーブしてください:

```bash
python -m scripts.chat_web
```

そして表示された URL にアクセスしてください。正しくアクセスするように注意してください。例えば Lambda では、自分がいるノードのパブリック IP に続けてポートを指定します。例えば [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/) のようになります。そして、いつも ChatGPT と話すように、あなたの LLM と話してください！ ストーリーや詩を書かせてみましょう。あなたが誰なのかを尋ねて、ハルシネーションを見てみましょう。なぜ空が青いのか尋ねてみましょう。あるいはなぜ緑なのか。このスピードランは 4e19 FLOPs の能力のモデルなので、幼稚園児と話しているような感じです :)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

いくつか補足:

- このコードは Ampere の 8XA100 GPU ノードでも問題なく動作しますが、少し遅くなります。
- すべてのコードは、`torchrun` を省略すれば単一の GPU でも問題なく動作し、ほぼ同一の結果を生成します（コードは自動的に勾配累積に切り替わります）が、8 倍長く待つ必要があります。
- お使いの GPU のメモリが 80GB 未満の場合、いくつかのハイパーパラメータを調整しないと OOM になる、つまり VRAM が不足します。スクリプト内の `--device-batch-size` を探し、収まるまで減らしてください。例えば 32（デフォルト）から 16、8、4、2、あるいは 1 まで。それより小さくする場合は、もう少し自分が何をしているか理解し、より創意工夫する必要があります。
- ほとんどのコードはかなり素朴な PyTorch なので、それをサポートするもの（xpu、mps など）なら何でも動作するはずですが、私自身がこれらのコードパスすべてを試したわけではないので、不具合があるかもしれません。

## 研究

もしあなたが研究者で nanochat の改善を手伝いたいなら、興味深い 2 つのスクリプトは [runs/scaling_laws.sh](runs/scaling_laws.sh) と [runs/miniseries.sh](runs/miniseries.sh) です。関連ドキュメントについては [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) を参照してください。素早い実験（約 5 分の事前学習の実行）に私のお気に入りのスケールは、12 レイヤーのモデル（GPT-1 サイズ）を訓練することです。例えば次のようにします:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

これは wandb（実行名 "d12"）を使用し、CORE 指標を最後のステップでのみ実行し、中間チェックポイントのサンプリングと保存を行いません。私はコードの何かを変更し、d12（または d16 など）を再実行して、それが役立ったかどうかを確認する、という反復ループが好きです。実行が役立つかどうかを確認するために、私は次の wandb プロットを監視するのが好きです:

1. `val_bpb`（語彙サイズ不変の単位であるビット毎バイトでの検証損失）を、`step`、`total_training_time`、`total_training_flops` の関数として。
2. `core_metric`（DCLM CORE スコア）
3. VRAM 使用率、`train/mfu`（モデル FLOPS 使用率）、`train/tok_per_sec`（訓練スループット）

例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照してください。

重要な点として、nanochat はたった 1 つの複雑さのダイヤル、すなわちトランスフォーマーの深さを中心に書かれ、構成されています。この単一の整数が、その他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、訓練ホライズン、重み減衰、…）を自動的に決定するため、訓練されたモデルは計算量最適な形で出来上がります。その狙いは、ユーザーがこれらを考えたり設定したりする必要がなく、単に `--depth` を使ってより小さなモデルやより大きなモデルを要求するだけで、すべてが「ただ動く」というものです。深さをスイープすることで、さまざまなサイズの計算量最適なモデルから成る nanochat ミニシリーズが得られます。GPT-2 相当の能力のモデル（現時点で最も関心が高いもの）は、現在のコードでは d24〜d26 のあたりに位置しています。しかし、リポジトリへのいかなる候補となる変更も、すべての深さの設定で機能するほど原理的でなければなりません。

## CPU / MPS での実行

スクリプト [runs/runcpu.sh](runs/runcpu.sh) は、CPU や Apple Silicon 上で実行する非常にシンプルな例を示しています。訓練されている LLM を劇的に縮小し、数十分程度の妥当な訓練時間に収まるようにします。この方法では強力な結果は得られません。

## 精度 / dtype

nanochat は `torch.amp.autocast` を使用しません。代わりに、精度は単一のグローバルな `COMPUTE_DTYPE`（`nanochat/common.py` で定義）を通じて明示的に管理されます。デフォルトでは、これはお使いのハードウェアに基づいて自動検出されます:

| ハードウェア | デフォルトの dtype | 理由 |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | ネイティブな bf16 テンソルコア |
| CUDA SM < 80 (V100, T4, ...) | `float32` | bf16 なし; fp16 は `NANOCHAT_DTYPE=float16` で利用可能（GradScaler を使用） |
| CPU / MPS | `float32` | 低精度テンソルコアなし |

`NANOCHAT_DTYPE` 環境変数を使ってデフォルトを上書きできます:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

仕組み: モデルの重みは（オプティマイザの精度のために）fp32 で保存されますが、私たちのカスタム `Linear` レイヤーがフォワードパス中にそれらを `COMPUTE_DTYPE` にキャストします。埋め込みはメモリを節約するために直接 `COMPUTE_DTYPE` で保存されます。これにより、autocast と同じ混合精度の恩恵を受けつつ、何をどの精度で実行するかを完全に明示的に制御できます。

注意: `float16` での訓練は、勾配のアンダーフローを防ぐために `base_train.py` で `GradScaler` を自動的に有効にします。SFT もこれをサポートしていますが、RL は現在サポートしていません。fp16 での推論はどこでも問題なく動作します。

## ガイド

役立つ情報を含むかもしれない多数のガイドを公開しています。新しいものから古いものの順に:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) は、最初の nanochat モデルのミニシリーズを記録しています。
- nanochat に新しい能力を追加するには、[ガイド: strawberry の r を数える（および能力を一般的に追加する方法）](https://github.com/karpathy/nanochat/discussions/164)を参照してください。
- あなたの nanochat をカスタマイズするには、Discussions の[ガイド: あなたの nanochat にアイデンティティを注入する](https://github.com/karpathy/nanochat/discussions/139)を参照してください。これは、合成データの生成とそのデータを SFT 段階に混ぜ込むことを通じて、あなたの nanochat の個性をどのように調整できるかを説明しています。
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) は nanochat を紹介していますが、今では一部に非推奨の情報が含まれており、モデルは現在の master よりもずっと古いものです（結果も劣ります）。

## ファイル構造

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # アイデンティティ用の合成データの例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 事前学習データのシャード生成
├── nanochat
│   ├── __init__.py                 # 空
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存/読み込み
│   ├── common.py                   # その他の小さなユーティリティ、生活の質の向上
│   ├── core_eval.py                # ベースモデルの CORE スコアを評価（DCLM 論文）
│   ├── dataloader.py               # トークン化分散データローダー
│   ├── dataset.py                  # 事前学習データのダウンロード/読み込みユーティリティ
│   ├── engine.py                   # KV Cache を用いた効率的なモデル推論
│   ├── execution.py                # LLM がツールとして Python コードを実行できるようにする
│   ├── gpt.py                      # GPT nn.Module トランスフォーマー
│   ├── logo.svg
│   ├── loss_eval.py                # ビット毎バイトを評価（損失の代わりに）
│   ├── optim.py                    # AdamW + Muon オプティマイザ、1GPU および分散
│   ├── report.py                   # nanochat レポート作成用のユーティリティ
│   ├── tokenizer.py                # GPT-4 スタイルの BPE トークナイザラッパー
│   └── ui.html                     # nanochat フロントエンド用の HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # ミニシリーズ訓練スクリプト
│   ├── runcpu.sh                   # CPU/MPS での実行方法の小さな例
│   ├── scaling_laws.sh             # スケーリング則の実験
│   └── speedrun.sh                 # 約 100 ドルの nanochat d20 を訓練
├── scripts
│   ├── base_eval.py                # ベースモデル: CORE スコア、ビット毎バイト、サンプル
│   ├── base_train.py               # ベースモデル: 訓練
│   ├── chat_cli.py                 # チャットモデル: CLI 経由で会話
│   ├── chat_eval.py                # チャットモデル: 評価タスク
│   ├── chat_rl.py                  # チャットモデル: 強化学習
│   ├── chat_sft.py                 # チャットモデル: SFT を訓練
│   ├── chat_web.py                 # チャットモデル: WebUI 経由で会話
│   ├── tok_eval.py                 # トークナイザ: 圧縮率を評価
│   └── tok_train.py                # トークナイザ: 訓練する
├── tasks
│   ├── arc.py                      # 多肢選択式の科学問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 任意の jsonl 会話から Task を作成
│   ├── gsm8k.py                    # 8K の小学校算数問題
│   ├── humaneval.py                # 名称は誤り; シンプルな Python コーディングタスク
│   ├── mmlu.py                     # 多肢選択式問題、幅広いトピック
│   ├── smoltalk.py                 # HF の SmolTalk を集約したデータセット
│   └── spellingbee.py              # モデルにスペルや文字数の数え方を教えるタスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## 貢献

nanochat の目標は、1000 ドル未満の予算でエンドツーエンドに扱えるマイクロモデルの最先端技術を向上させることです。アクセシビリティは全体的なコストに関するものですが、認知的な複雑さに関するものでもあります。nanochat は、徹底的に構成可能な LLM「フレームワーク」ではありません。コードベースには、巨大な設定オブジェクトも、モデルファクトリも、if-then-else の怪物もありません。これは、最初から最後まで実行して、あなたが会話できる ChatGPT モデルを生成するように設計された、単一の、まとまりのある、最小限の、読みやすく、ハック可能で、最大限フォークしやすい「強力なベースライン」コードベースです。現在、私個人にとって最も興味深い部分は、GPT-2 への遅延（すなわち 0.256525 を上回る CORE スコアを得ること）を高速化することです。現在これには約 3 時間かかりますが、事前学習段階を改善することでさらに改善できます。

現在の AI ポリシー: 開示。PR を提出する際は、LLM が大きく貢献した部分で、あなたが書いていない、または完全には理解していない部分があれば、それを申告してください。

## 謝辞

- 名前（nanochat）は、事前学習のみをカバーしていた私の以前のプロジェクト [nanoGPT](https://github.com/karpathy/nanoGPT) に由来します。
- nanochat はまた、明確な指標とリーダーボードで nanoGPT リポジトリをゲーム化した [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) にも触発されており、その多くのアイデアと事前学習用の一部の実装を借用しています。
- fineweb と smoltalk を提供してくれた [HuggingFace](https://huggingface.co/) に感謝します。
- このプロジェクトの開発に使用した計算資源を提供してくれた [Lambda](https://lambda.ai/service/gpu-cloud) に感謝します。
- 助言と指導をくれた、LLM ささやきの達人 🧙‍♂️ Alec Radford に感謝します。
- nanochat の issue、プルリクエスト、ディスカッションの管理を手伝ってくれた、リポジトリの皇帝 Sofie [@svlandeg](https://github.com/svlandeg) に感謝します。

## 引用

研究で nanochat が役立つと感じた場合は、単純に次のように引用してください:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MIT
