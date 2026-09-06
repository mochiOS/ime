import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from common import (
	ImeStudent,
	ImeTeacher,
	RankingDataset,
	collate_rank_only,
	distillation_loss,
	listwise_loss,
	ranking_metrics,
	tokenize,
)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input", required=True)
	parser.add_argument("--teacher", required=True)
	parser.add_argument("--output", required=True)

	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=5e-5)
	parser.add_argument("--eval-ratio", type=float, default=0.1)
	parser.add_argument("--max-length", type=int, default=96)
	parser.add_argument("--temperature", type=float, default=2.0)
	parser.add_argument("--distill-weight", type=float, default=0.7)
	parser.add_argument("--seed", type=int, default=42)

	return parser.parse_args()


def evaluate(
	model,
	tokenizer,
	loader,
	device,
	max_length,
):
	model.eval()

	total_groups = 0
	top1_sum = 0.0
	mrr_sum = 0.0

	with torch.inference_mode():
		for batch in loader:
			inputs = tokenize(
				tokenizer,
				batch,
				device,
				max_length,
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

	model.train()

	return {
		"top1": top1_sum / max(total_groups, 1),
		"mrr": mrr_sum / max(total_groups, 1),
	}


def main():
	args = parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	rows = pq.read_table(
		args.input
	).to_pylist()

	if len(rows) < 2:
		raise ValueError(
			"dataset must contain at least 2 groups"
		)

	random.Random(args.seed).shuffle(rows)

	eval_count = max(
		1,
		int(len(rows) * args.eval_ratio),
	)

	eval_count = min(
		eval_count,
		len(rows) - 1,
	)

	eval_rows = rows[:eval_count]
	train_rows = rows[eval_count:]

	train_loader = DataLoader(
		RankingDataset(train_rows),
		batch_size=args.batch_size,
		shuffle=True,
		collate_fn=collate_rank_only,
	)

	eval_loader = DataLoader(
		RankingDataset(eval_rows),
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_rank_only,
	)

	device = torch.device(
		"cuda"
		if torch.cuda.is_available()
		else "cpu"
	)

	if device.type == "cuda":
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	tokenizer = AutoTokenizer.from_pretrained(
		args.teacher,
		use_fast=True,
	)

	teacher = ImeTeacher.load(
		args.teacher,
		device,
	)

	teacher.eval()

	for parameter in teacher.parameters():
		parameter.requires_grad_(False)

	student = ImeStudent.create(
		vocab_size=len(tokenizer),
		max_length=args.max_length,
	).to(device)

	optimizer = AdamW(
		student.parameters(),
		lr=args.learning_rate,
		weight_decay=0.01,
	)

	output = Path(args.output)
	output.mkdir(
		parents=True,
		exist_ok=True,
	)

	best_mrr = -1.0

	for epoch in range(args.epochs):
		student.train()
		running_loss = 0.0

		for batch in tqdm(
			train_loader,
			desc=(
				f"epoch "
				f"{epoch + 1}/"
				f"{args.epochs}"
			),
		):
			inputs = tokenize(
				tokenizer,
				batch,
				device,
				args.max_length,
			)

			with torch.inference_mode():
				teacher_scores = teacher(
					**inputs
				)["rank"].float()

			student_scores = student(
				**inputs
			).float()

			hard = listwise_loss(
				student_scores,
				batch["group_sizes"],
			)

			soft = distillation_loss(
				student_scores,
				teacher_scores,
				batch["group_sizes"],
				args.temperature,
			)

			loss = (
				(1.0 - args.distill_weight)
				* hard
				+ args.distill_weight
				* soft
			)

			if not torch.isfinite(loss):
				raise RuntimeError(
					"non-finite student loss"
				)

			optimizer.zero_grad(
				set_to_none=True
			)

			loss.backward()

			torch.nn.utils.clip_grad_norm_(
				student.parameters(),
				1.0,
			)

			optimizer.step()

			running_loss += loss.item()

		metrics = evaluate(
			student,
			tokenizer,
			eval_loader,
			device,
			args.max_length,
		)

		print(
			f"epoch {epoch + 1}: "
			f"loss="
			f"{running_loss / max(len(train_loader), 1):.6f} "
			f"top1={metrics['top1']:.4f} "
			f"mrr={metrics['mrr']:.4f}"
		)

		if metrics["mrr"] > best_mrr:
			best_mrr = metrics["mrr"]

			student.save(output)

			tokenizer.save_pretrained(
				output
			)

			with (
				output / "metrics.json"
			).open(
				"w",
				encoding="utf-8",
			) as file:
				json.dump(
					{
						"epoch": epoch + 1,
						**metrics,
					},
					file,
					ensure_ascii=False,
					indent=2,
				)


if __name__ == "__main__":
	main()
