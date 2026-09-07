import argparse
from pathlib import Path

from sudachipy import Dictionary
from tqdm.auto import tqdm


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--input",
		required=True,
	)

	parser.add_argument(
		"--output",
		required=True,
	)

	return parser.parse_args()


def main():
	args = parse_args()

	tokenizer = Dictionary().create()

	input_path = Path(args.input)
	output_path = Path(args.output)

	output_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	with input_path.open(
		encoding="utf-8",
	) as source:
		with output_path.open(
			"w",
			encoding="utf-8",
			newline="\n",
		) as output:
			for line in tqdm(source):
				sentence = line.strip()

				if not sentence:
					continue

				words = []

				for morpheme in tokenizer.tokenize(
					sentence
				):
					surface = (
						morpheme
						.surface()
						.replace("\t", " ")
						.replace("\r", "")
						.replace("\n", "")
					)

					if surface:
						words.append(surface)

				if not words:
					continue

				output.write(
					"\t".join(words)
				)

				output.write("\n")


if __name__ == "__main__":
	main()