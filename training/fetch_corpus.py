import argparse
import re
from pathlib import Path

from datasets import load_dataset


DATASET = "hotchpotch/fineweb-2-edu-japanese"
CONFIG = "sample_10BT"

SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])")
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
URL = re.compile(r"https?://|www\.", re.IGNORECASE)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", required=True)
	parser.add_argument("--max-sentences", type=int, default=100000)
	return parser.parse_args()


def split_sentences(text):
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = re.sub(r"[ \t]+", " ", text)

	for paragraph in re.split(r"\n+", text):
		paragraph = paragraph.strip()

		if not paragraph:
			continue

		for sentence in SENTENCE_SPLIT.split(paragraph):
			sentence = sentence.strip()

			if sentence:
				yield sentence


def usable(sentence):
	length = len(sentence)

	if length < 5 or length > 160:
		return False

	if URL.search(sentence):
		return False

	japanese = len(JAPANESE.findall(sentence))

	if japanese < 3:
		return False

	return japanese / max(length, 1) >= 0.5


def main():
	args = parse_args()

	output = Path(args.output)
	output.parent.mkdir(parents=True, exist_ok=True)

	dataset = load_dataset(
		DATASET,
		CONFIG,
		split="train",
		streaming=True,
	)

	count = 0

	with output.open("w", encoding="utf-8", newline="\n") as out:
		for row in dataset:
			text = row.get("text")

			if not isinstance(text, str):
				continue

			for sentence in split_sentences(text):
				if not usable(sentence):
					continue

				out.write(sentence + "\n")
				count += 1

				if count >= args.max_sentences:
					print(f"sentences: {count}")
					return


if __name__ == "__main__":
	main()
