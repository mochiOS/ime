import argparse
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "hotchpotch/fineweb-2-edu-japanese"
DEFAULT_CONFIG = "sample_10BT"

DEFAULT_SHARD_SIZE = 1_000_000
DEFAULT_MIN_LENGTH = 5
DEFAULT_MAX_LENGTH = 200
DEFAULT_MIN_SCORE = 2.5

SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])")
JAPANESE_PATTERN = re.compile(
	r"[\u3040-\u30ff\u3400-\u9fff]"
)
URL_PATTERN = re.compile(
	r"https?://|www\.",
	re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--output-dir",
		default="training/corpus",
	)

	parser.add_argument(
		"--config",
		default=DEFAULT_CONFIG,
	)

	parser.add_argument(
		"--shard-size",
		type=int,
		default=DEFAULT_SHARD_SIZE,
	)

	parser.add_argument(
		"--min-length",
		type=int,
		default=DEFAULT_MIN_LENGTH,
	)

	parser.add_argument(
		"--max-length",
		type=int,
		default=DEFAULT_MAX_LENGTH,
	)

	parser.add_argument(
		"--min-score",
		type=float,
		default=DEFAULT_MIN_SCORE,
	)

	parser.add_argument(
		"--max-sentences",
		type=int,
		default=None,
	)

	return parser.parse_args()


def normalize_text(text: str) -> str:
	text = text.replace("\r\n", "\n")
	text = text.replace("\r", "\n")

	text = re.sub(
		r"[ \t]+",
		" ",
		text,
	)

	return text


def split_sentences(text: str) -> list[str]:
	text = normalize_text(text)

	sentences = []

	for paragraph in re.split(r"\n+", text):
		paragraph = paragraph.strip()

		if not paragraph:
			continue

		for sentence in SENTENCE_PATTERN.split(paragraph):
			sentence = sentence.strip()

			if sentence:
				sentences.append(sentence)

	return sentences


def is_usable_sentence(
	sentence: str,
	min_length: int,
	max_length: int,
) -> bool:
	length = len(sentence)

	if length < min_length:
		return False

	if length > max_length:
		return False

	if URL_PATTERN.search(sentence):
		return False

	japanese_count = len(
		JAPANESE_PATTERN.findall(sentence)
	)

	if japanese_count < 3:
		return False

	japanese_ratio = (
		japanese_count
		/ max(length, 1)
	)

	if japanese_ratio < 0.3:
		return False

	return True


class ShardWriter:
	def __init__(
		self,
		output_dir: Path,
		shard_size: int,
	):
		self.output_dir = output_dir
		self.shard_size = shard_size

		self.shard_index = 0
		self.shard_count = 0
		self.total_count = 0

		self.file = None

		self.output_dir.mkdir(
			parents=True,
			exist_ok=True,
		)

	def open_next(self) -> None:
		if self.file is not None:
			self.file.close()

		path = self.output_dir / (
			f"part-{self.shard_index:05d}.txt"
		)

		print(f"opening {path}")

		self.file = path.open(
			"w",
			encoding="utf-8",
			newline="\n",
		)

		self.shard_index += 1
		self.shard_count = 0

	def write(self, sentence: str) -> None:
		if (
			self.file is None
			or self.shard_count >= self.shard_size
		):
			self.open_next()

		self.file.write(sentence)
		self.file.write("\n")

		self.shard_count += 1
		self.total_count += 1

	def close(self) -> None:
		if self.file is not None:
			self.file.close()
			self.file = None


def main() -> None:
	args = parse_args()

	if args.shard_size <= 0:
		raise ValueError(
			"--shard-size must be greater than zero"
		)

	output_dir = Path(args.output_dir)

	print(f"dataset: {DATASET_NAME}")
	print(f"config: {args.config}")
	print(f"output: {output_dir}")
	print("streaming dataset...")

	dataset = load_dataset(
		DATASET_NAME,
		args.config,
		split="train",
		streaming=True,
	)

	writer = ShardWriter(
		output_dir,
		args.shard_size,
	)

	document_count = 0

	try:
		for row in dataset:
			document_count += 1

			score = row.get("score")

			if (
				score is not None
				and float(score) < args.min_score
			):
				continue

			text = row.get("text")

			if not isinstance(text, str):
				continue

			for sentence in split_sentences(text):
				if not is_usable_sentence(
					sentence,
					args.min_length,
					args.max_length,
				):
					continue

				writer.write(sentence)

				if (
					args.max_sentences is not None
					and writer.total_count
					>= args.max_sentences
				):
					return

			if document_count % 10_000 == 0:
				print(
					f"documents={document_count:,} "
					f"sentences={writer.total_count:,}"
				)

	finally:
		writer.close()

	print(
		f"documents: {document_count:,}"
	)

	print(
		f"sentences: {writer.total_count:,}"
	)


if __name__ == "__main__":
	main()