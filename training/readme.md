このディレクトリにはIMEのニューラルモデルを学習するためのスクリプトを置いています。
学習済みモデルは`engine`が生成した変換候補を文脈に基づいて並び替えるために使用します。

## Requirements

Rust側を先にビルドしてください。

```sh
cargo build
```

学習データ生成にはPerlとPythonが必要です。

Pythonの依存パッケージ:

```sh
pip install sudachipy sudachidict_core
```

モデル学習には以下も必要です。

```sh
pip install torch transformers sentencepiece
```

GPU が利用可能な場合、`train.py`は自動的にCUDAを使用します。

## Dictionary

先に日本語辞書をダウンロードしてください。

```sh
vendor/ja/download.sh
```

Windows の場合:

```bat
vendor\ja\download.bat
```

その後、SudachiDictから`ja.mime`（.mimeはmochiOS IMEの独自辞書フォーマットです）を生成します。

```sh
cargo run -p engine --bin mimec -- --lex vendor/ja/small_lex.csv --lex vendor/ja/core_lex.csv --matrix vendor/ja/matrix.def -o vendor/ja/ja.mime
```

## Corpus

学習元となる日本語文章をUTF-8のテキストファイルとして用意します。

1行につき1文です。

例:

```text
今日はいい天気ですね。
私は学校へ行きました。
空がとてもきれいです。
明日は雨が降るかもしれません。
```

例として、

```text
training/corpus.txt
```

に保存します。

## Generate Dataset

`generate_dataset.pl`は正解文から読みを生成し、その読みを`engine`に入力して誤変換候補を作ります。

実行例:

```sh
perl training/generate_dataset.pl \
    --input training/corpus.txt \
    --output training/dataset.tsv \
    --engine target/debug/candidates \
    --dictionary vendor/ja/ja.mime \
    --reading-script training/reading.py \
    --limit 32
```

Windows PowerShellでは1で実行できます。

```powershell
perl training/generate_dataset.pl --input training/corpus.txt --output training/dataset.tsv --engine target/debug/candidates.exe --dictionary vendor/ja/ja.mime --reading-script training/reading.py --limit 32
```

`--limit`は1つの読みから生成する変換候補数です。

通常は`32`を使用してください。

## Train

学習は以下で開始できます。

```sh
python training/train.py \
    --input training/dataset.tsv \
    --output training/model
```

Windows PowerShell:

```powershell
python training/train.py --input training/dataset.tsv --output training/model
```

デフォルトではTeacherとStudentの両方を学習します。

TeacherをIMEに直接搭載するのではなく、Teacherのランキング能力をStudentに蒸留します。

実際のIMEではStudentのみ使用します。
