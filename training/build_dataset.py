import argparse
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
from sudachipy import Dictionary
from tqdm.auto import tqdm


def parse_args():
\tparser = argparse.ArgumentParser()
\tparser.add_argument("--input", required=True)
\tparser.add_argument("--output", required=True)
\tparser.add_argument("--engine", required=True)
\tparser.add_argument("--dictionary", required=True)
\tparser.add_argument("--limit", type=int, default=16)
\treturn parser.parse_args()


def katakana_to_hiragana(text):
\treturn "".join(
\t\tchr(ord(ch) - 0x60)
\t\tif "\u30a1" <= ch <= "\u30f6"
\t\telse ch
\t\tfor ch in text
\t)


class CandidateEngine:
\tdef __init__(self, executable, dictionary, limit):
\t\tself.process = subprocess.Popen(
\t\t\t[
\t\t\t\texecutable,
\t\t\t\t"--dictionary",
\t\t\t\tdictionary,
\t\t\t\t"--limit",
\t\t\t\tstr(limit),
\t\t\t],
\t\t\tstdin=subprocess.PIPE,
\t\t\tstdout=subprocess.PIPE,
\t\t\tstderr=subprocess.PIPE,
\t\t\ttext=True,
\t\t\tencoding="utf-8",
\t\t\tbufsize=1,
\t\t)

\tdef candidates(self, reading):
\t\tself.process.stdin.write(reading + "\n")
\t\tself.process.stdin.flush()

\t\tcount_line = self.process.stdout.readline()

\t\tif not count_line:
\t\t\terror = self.process.stderr.read()
\t\t\traise RuntimeError(error)

\t\tcount = int(count_line.strip())

\t\treturn [
\t\t\tself.process.stdout.readline().rstrip("\r\n")
\t\t\tfor _ in range(count)
\t\t]

\tdef close(self):
\t\tif self.process.poll() is None:
\t\t\tself.process.stdin.close()
\t\t\tself.process.wait()


def infer_error_type(positive, candidate):
\tif positive == candidate:
\t\treturn 0

\tif any("\u30a0" <= ch <= "\u30ff" for ch in candidate):
\t\treturn 3

\tif len(candidate) != len(positive):
\t\treturn 2

\tif any(ch.isascii() and ch.isalnum() for ch in candidate):
\t\treturn 4

\treturn 1


def main():
\targs = parse_args()
\tsudachi = Dictionary().create()

\twith open(args.input, encoding="utf-8") as file:
\t\tsentences = [
\t\t\tline.rstrip("\r\n")
\t\t\tfor line in file
\t\t\tif line.strip()
\t\t]

\tengine = CandidateEngine(
\t\targs.engine,
\t\targs.dictionary,
\t\targs.limit,
\t)

\trows = []

\ttry:
\t\tfor positive in tqdm(sentences):
\t\t\treading = katakana_to_hiragana(
\t\t\t\t"".join(
\t\t\t\t\tm.reading_form()
\t\t\t\t\tfor m in sudachi.tokenize(positive)
\t\t\t\t)
\t\t\t)

\t\t\tif not reading:
\t\t\t\tcontinue

\t\t\tcandidates = engine.candidates(reading)

\t\t\tseen = {positive}
\t\t\tnegatives = []

\t\t\tfor candidate in candidates:
\t\t\t\tif candidate in seen:
\t\t\t\t\tcontinue

\t\t\t\tseen.add(candidate)
\t\t\t\tnegatives.append({
\t\t\t\t\t"text": candidate,
\t\t\t\t\t"error_type": infer_error_type(positive, candidate),
\t\t\t\t})

\t\t\tif negatives:
\t\t\t\trows.append({
\t\t\t\t\t"reading": reading,
\t\t\t\t\t"positive": positive,
\t\t\t\t\t"negatives": negatives,
\t\t\t\t})
\tfinally:
\t\tengine.close()

\ttable = pa.Table.from_pylist(rows)
\tpq.write_table(table, args.output)
\tprint(f"groups: {len(rows)}")


if __name__ == "__main__":
\tmain()
