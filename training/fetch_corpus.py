import argparse
import re
from datasets import load_dataset

DATASET = "hotchpotch/fineweb-2-edu-japanese"
CONFIG = "sample_10BT"

SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])")
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
URL = re.compile(r"https?://|www\.", re.IGNORECASE)


def parse_args():
\tparser = argparse.ArgumentParser()
\tparser.add_argument("--output", required=True)
\tparser.add_argument("--max-sentences", type=int, default=100000)
\treturn parser.parse_args()


def split_sentences(text):
\ttext = text.replace("\r\n", "\n").replace("\r", "\n")
\ttext = re.sub(r"[ \t]+", " ", text)

\tfor paragraph in re.split(r"\n+", text):
\t\tparagraph = paragraph.strip()

\t\tif not paragraph:
\t\t\tcontinue

\t\tfor sentence in SENTENCE_SPLIT.split(paragraph):
\t\t\tsentence = sentence.strip()

\t\t\tif sentence:
\t\t\t\tyield sentence


def usable(sentence):
\tlength = len(sentence)

\tif length < 5 or length > 160:
\t\treturn False

\tif URL.search(sentence):
\t\treturn False

\tjapanese = len(JAPANESE.findall(sentence))

\tif japanese < 3:
\t\treturn False

\treturn japanese / max(length, 1) >= 0.5


def main():
\targs = parse_args()

\tdataset = load_dataset(
\t\tDATASET,
\t\tCONFIG,
\t\tsplit="train",
\t\tstreaming=True,
\t)

\tcount = 0

\twith open(args.output, "w", encoding="utf-8", newline="\n") as out:
\t\tfor row in dataset:
\t\t\ttext = row.get("text")

\t\t\tif not isinstance(text, str):
\t\t\t\tcontinue

\t\t\tfor sentence in split_sentences(text):
\t\t\t\tif not usable(sentence):
\t\t\t\t\tcontinue

\t\t\t\tout.write(sentence + "\n")
\t\t\t\tcount += 1

\t\t\t\tif count >= args.max_sentences:
\t\t\t\t\tprint(f"sentences: {count}")
\t\t\t\t\treturn


if __name__ == "__main__":
\tmain()
