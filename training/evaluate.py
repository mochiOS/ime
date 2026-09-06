import argparse

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import (
	ImeStudent,
	RankingDataset,
	collate_rank_only,
	ranking_metrics,
	tokenize,
)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--max-length", type=int, default=96)
	return parser.parse_args()


def main():
	args = parse_args()

	rows = pq.read_table(
		args.input
	).to_pylist()

	device = torch.device(
		"cuda"
		if torch.cuda.is_available()
		else "cpu"
	)

	tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		use_fast=True,
	)

	model = ImeStudent.load(
		args.model,
		device,
	)

	model.eval()

	loader = DataLoader(
		RankingDataset(rows),
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_rank_only,
	)

	total_groups = 0
	top1_sum = 0.0
	mrr_sum = 0.0

	with torch.inference_mode():
		for batch in loader:
			inputs = tokenize(
				tokenizer,
				batch,
				device,
				args.max_length,
			)

			scores = model(
				**inputs
			).float()

			metrics = ranking_metrics(
				scores,
				batch["group_sizes"],
			)

			group_count = len(
				batch["group_sizes"]
			)

			top1_sum += (
				metrics["top1"]
				* group_count
			)

			mrr_sum += (
				metrics["mrr"]
				* group_count
			)

			total_groups += group_count

	print(f"groups: {total_groups}")
	print(
		f"top1: "
		f"{top1_sum / max(total_groups, 1):.6f}"
	)
	print(
		f"mrr: "
		f"{mrr_sum / max(total_groups, 1):.6f}"
	)


if __name__ == "__main__":
	main()
