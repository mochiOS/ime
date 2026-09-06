import argparse
import pyarrow.parquet as pq


def parse_args():
\tparser = argparse.ArgumentParser()
\tparser.add_argument("--input", required=True)
\tparser.add_argument("--model", required=True)
\treturn parser.parse_args()


def main():
\targs = parse_args()
\trows = pq.read_table(args.input).to_pylist()

\tprint(f"groups: {len(rows)}")
\tprint("Model evaluation runner will calculate Top-1 and MRR.")


if __name__ == "__main__":
\tmain()
