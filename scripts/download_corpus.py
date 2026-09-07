import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi
from tqdm.auto import tqdm


REPOSITORY = "hotchpotch/fineweb-2-edu-japanese"
CONFIG_DIRECTORY = "sample_10BT"

BASE_URL = (
	"https://huggingface.co/datasets/"
	f"{REPOSITORY}/resolve/main"
)

DEFAULT_OUTPUT = "vendor/corpus/corpus.txt"
DEFAULT_CACHE = "target/scripts/cache"

DEFAULT_MIN_LENGTH = 5
DEFAULT_MAX_LENGTH = 160

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

SENTENCE_SPLIT = re.compile(
	r"(?<=[。！？!?])"
)

JAPANESE = re.compile(
	r"[\u3040-\u30ff\u3400-\u9fff]"
)

URL = re.compile(
	r"https?://|www\.",
	re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--output",
		default=DEFAULT_OUTPUT,
	)

	parser.add_argument(
		"--cache",
		default=DEFAULT_CACHE,
	)

	parser.add_argument(
		"--max-sentences",
		type=int,
		default=None,
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
		"--keep-cache",
		action="store_true",
	)

	return parser.parse_args()


def get_parquet_files() -> list[str]:
	api = HfApi()

	files = api.list_repo_files(
		REPOSITORY,
		repo_type="dataset",
	)

	files = [
		path
		for path in files
		if path.startswith(
			f"{CONFIG_DIRECTORY}/"
		)
		and path.endswith(".parquet")
	]

	files.sort()

	if not files:
		raise RuntimeError(
			"no Parquet files were found"
		)

	return files


def download_file(
	remote_path: str,
	local_path: Path,
) -> None:
	url = (
		f"{BASE_URL}/{remote_path}"
		"?download=true"
	)

	local_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	temporary = local_path.with_suffix(
		local_path.suffix + ".part"
	)

	if local_path.exists():
		return

	if temporary.exists():
		temporary.unlink()

	print(
		f"downloading {remote_path}"
	)

	with requests.get(
		url,
		stream=True,
		timeout=(30, 300),
	) as response:
		response.raise_for_status()

		total = int(
			response.headers.get(
				"content-length",
				0,
			)
		)

		with temporary.open("wb") as file:
			with tqdm(
				total=total,
				unit="B",
				unit_scale=True,
				unit_divisor=1024,
				desc=local_path.name,
			) as progress:
				for chunk in response.iter_content(
					chunk_size=DOWNLOAD_CHUNK_SIZE,
				):
					if not chunk:
						continue

					file.write(chunk)

					progress.update(
						len(chunk)
					)

	temporary.replace(local_path)


def split_sentences(text: str):
	text = text.replace(
		"\r\n",
		"\n",
	)

	text = text.replace(
		"\r",
		"\n",
	)

	text = re.sub(
		r"[ \t]+",
		" ",
		text,
	)

	for paragraph in re.split(
		r"\n+",
		text,
	):
		paragraph = paragraph.strip()

		if not paragraph:
			continue

		for sentence in SENTENCE_SPLIT.split(
			paragraph
		):
			sentence = sentence.strip()

			if sentence:
				yield sentence


def usable_sentence(
	sentence: str,
	min_length: int,
	max_length: int,
) -> bool:
	length = len(sentence)

	if length < min_length:
		return False

	if length > max_length:
		return False

	if URL.search(sentence):
		return False

	japanese_count = len(
		JAPANESE.findall(sentence)
	)

	if japanese_count < 3:
		return False

	if (
		japanese_count
		/ max(length, 1)
		< 0.5
	):
		return False

	return True


def process_parquet(
	path: Path,
	output,
	min_length: int,
	max_length: int,
	current_count: int,
	max_sentences: int | None,
) -> tuple[int, bool]:
	parquet = pq.ParquetFile(path)

	for batch in parquet.iter_batches(
		batch_size=4096,
		columns=["text"],
	):
		text_index = batch.schema.get_field_index(
			"text"
		)

		column = batch.column(
			text_index
		)

		for value in column:
			if not value.is_valid:
				continue

			text = value.as_py()

			if not isinstance(text, str):
				continue

			for sentence in split_sentences(
				text
			):
				if not usable_sentence(
					sentence,
					min_length,
					max_length,
				):
					continue

				output.write(sentence)
				output.write("\n")

				current_count += 1

				if (
					max_sentences is not None
					and current_count
					>= max_sentences
				):
					return (
						current_count,
						False,
					)

	return current_count, True


def main() -> None:
	args = parse_args()

	output_path = Path(
		args.output
	)

	cache_path = Path(
		args.cache
	)

	output_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	cache_path.mkdir(
		parents=True,
		exist_ok=True,
	)

	parquet_files = get_parquet_files()

	print(
		f"Parquet files: "
		f"{len(parquet_files)}"
	)

	sentence_count = 0

	with output_path.open(
		"w",
		encoding="utf-8",
		newline="\n",
	) as output:
		for index, remote_path in enumerate(
			parquet_files,
			start=1,
		):
			name = Path(
				remote_path
			).name

			local_path = (
				cache_path / name
			)

			print(
				f"[{index}/"
				f"{len(parquet_files)}] "
				f"{name}"
			)

			download_file(
				remote_path,
				local_path,
			)

			sentence_count, should_continue = (
				process_parquet(
					local_path,
					output,
					args.min_length,
					args.max_length,
					sentence_count,
					args.max_sentences,
				)
			)

			output.flush()

			print(
				f"sentences: "
				f"{sentence_count:,}"
			)

			if not args.keep_cache:
				local_path.unlink(
					missing_ok=True
				)

			if not should_continue:
				break

	print(
		f"done: {sentence_count:,} sentences"
	)

	print(
		f"output: {output_path}"
	)


if __name__ == "__main__":
	main()