import argparse
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sudachipy import Dictionary
from tqdm.auto import tqdm


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True)
	parser.add_argument("--output", required=True)
	parser.add_argument("--engine", required=True)
	parser.add_argument("--dictionary", required=True)
	parser.add_argument("--limit", type=int, default=16)
	return parser.parse_args()


def katakana_to_hiragana(text):
	return "".join(
		chr(ord(ch) - 0x60)
		if "\u30a1" <= ch <= "\u30f6"
		else ch
		for ch in text
	)


class CandidateEngine:
	def __init__(self, executable, dictionary, limit):
		self.process = subprocess.Popen(
			[
				executable,
				"--dictionary",
				dictionary,
				"--limit",
				str(limit),
			],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			encoding="utf-8",
			bufsize=1,
		)

	def candidates(self, reading):
		if self.process.poll() is not None:
			raise RuntimeError(self.process.stderr.read())

		self.process.stdin.write(reading + "\n")
		self.process.stdin.flush()

		count_line = self.process.stdout.readline()

		if not count_line:
			raise RuntimeError(self.process.stderr.read())

		count = int(count_line.strip())

		result = []

		for _ in range(count):
			line = self.process.stdout.readline()

			if not line:
				raise RuntimeError(
					"candidate engine terminated unexpectedly"
				)

			result.append(line.rstrip("\r\n"))

		return result

	def close(self):
		if self.process.poll() is None:
			self.process.stdin.close()
			self.process.wait()


def infer_error_type(positive, candidate):
	if positive == candidate:
		return 0

	if any("\u30a0" <= ch <= "\u30ff" for ch in candidate):
		return 3

	if any(ch.isascii() and ch.isalnum() for ch in candidate):
		return 4

	if len(candidate) != len(positive):
		return 2

	return 1


def main():
	args = parse_args()

	sudachi = Dictionary().create()

	with open(args.input, encoding="utf-8") as file:
		sentences = [
			line.rstrip("\r\n")
			for line in file
			if line.strip()
		]

	engine = CandidateEngine(
		args.engine,
		args.dictionary,
		args.limit,
	)

	rows = []

	try:
		for positive in tqdm(sentences):
			reading = katakana_to_hiragana(
				"".join(
					morpheme.reading_form()
					for morpheme in sudachi.tokenize(positive)
				)
			)

			if not reading:
				continue

			candidates = engine.candidates(reading)

			seen = {positive}
			negatives = []

			for candidate in candidates:
				if candidate in seen:
					continue

				seen.add(candidate)

				negatives.append(
					{
						"text": candidate,
						"error_type": infer_error_type(
							positive,
							candidate,
						),
					}
				)

			if not negatives:
				continue

			rows.append(
				{
					"reading": reading,
					"positive": positive,
					"negatives": negatives,
				}
			)
	finally:
		engine.close()

	output = Path(args.output)
	output.parent.mkdir(parents=True, exist_ok=True)

	table = pa.Table.from_pylist(rows)
	pq.write_table(table, output)

	print(f"groups: {len(rows)}")


if __name__ == "__main__":
	main()
