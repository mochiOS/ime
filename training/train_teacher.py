import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common import (
	ImeTeacher,
	RankingDataset,
	collate_groups,
	listwise_loss,
	pairwise_margin_loss,
	ranking_metrics,
	tokenize,
)


DEFAULT_MODEL = "ku-nlp/deberta-v3-base-japanese"

RANK_WEIGHT = 1.0
CORRECTNESS_WEIGHT = 0.3
PAIRWISE_WEIGHT = 0.2
ERROR_WEIGHT = 0.15


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input", required=True)
	parser.add_argument("--output", required=True)

	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--gradient-accumulation", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=1e-5)
	parser.add_argument("--eval-ratio", type=float, default=0.1)
	parser.add_argument("--max-length", type=int, default=96)
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

	total_loss = 0.0
	batches = 0

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

			output = model(**inputs)
			rank = output["rank"].float()

			loss = listwise_loss(
				rank,
				batch["group_sizes"],
			)

			metrics = ranking_metrics(
				rank,
				batch["group_sizes"],
			)

			group_count = len(
				batch["group_sizes"]
			)

			total_loss += loss.item()
			batches += 1

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
		"loss": total_loss / max(batches, 1),
		"top1": top1_sum / max(total_groups, 1),
		"mrr": mrr_sum / max(total_groups, 1),
	}


def get_device():
	try:
		import torch_directml

		device = torch_directml.device()

		print(f"device: {device}")
		print("backend: DirectML")

		return device

	except ImportError:
		device = torch.device(
			"cuda"
			if torch.cuda.is_available()
			else "cpu"
		)

		print(f"device: {device}")

		return device

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
		collate_fn=collate_groups,
	)

	eval_loader = DataLoader(
		RankingDataset(eval_rows),
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_groups,
	)

	device = get_device()

	if device.type == "cuda":
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	print(f"device: {device}")

	tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		use_fast=True,
	)

	model = ImeTeacher(
		model_name=args.model
	).to(device)

	optimizer = AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=0.01,
		eps=1e-6,
	)

	updates_per_epoch = (
		len(train_loader)
		+ args.gradient_accumulation - 1
	) // args.gradient_accumulation

	total_updates = (
		updates_per_epoch
		* args.epochs
	)

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=int(
			total_updates * 0.1
		),
		num_training_steps=total_updates,
	)

	output = Path(args.output)
	output.mkdir(
		parents=True,
		exist_ok=True,
	)

	best_mrr = -1.0

	for epoch in range(args.epochs):
		model.train()

		running_loss = 0.0
		step_count = 0

		optimizer.zero_grad(
			set_to_none=True
		)

		for step, batch in enumerate(
			tqdm(
				train_loader,
				desc=(
					f"epoch "
					f"{epoch + 1}/"
					f"{args.epochs}"
				),
			),
			start=1,
		):
			inputs = tokenize(
				tokenizer,
				batch,
				device,
				args.max_length,
			)

			labels = batch["labels"].to(
				device
			)

			error_types = batch[
				"error_types"
			].to(device)

			result = model(**inputs)

			rank_loss = listwise_loss(
				result["rank"],
				batch["group_sizes"],
			)

			correctness_loss = (
				F.binary_cross_entropy_with_logits(
					result["correctness"].float(),
					labels,
				)
			)

			margin_loss = pairwise_margin_loss(
				result["rank"],
				batch["group_sizes"],
			)

			error_loss = F.cross_entropy(
				result["error"].float(),
				error_types,
			)

			loss = (
				RANK_WEIGHT
				* rank_loss
				+ CORRECTNESS_WEIGHT
				* correctness_loss
				+ PAIRWISE_WEIGHT
				* margin_loss
				+ ERROR_WEIGHT
				* error_loss
			)

			if not torch.isfinite(loss):
				raise RuntimeError(
					f"non-finite loss: "
					f"epoch={epoch + 1} "
					f"step={step}"
				)

			unscaled_loss = (
				loss.detach().item()
			)

			(
				loss
				/ args.gradient_accumulation
			).backward()

			running_loss += unscaled_loss
			step_count += 1

			if (
				step
				% args.gradient_accumulation
				== 0
				or step == len(train_loader)
			):
				grad_norm = (
					torch.nn.utils
					.clip_grad_norm_(
						model.parameters(),
						0.5,
					)
				)

				if not torch.isfinite(
					grad_norm
				):
					raise RuntimeError(
						"non-finite gradient"
					)

				optimizer.step()

				optimizer.zero_grad(
					set_to_none=True
				)

				scheduler.step()

		metrics = evaluate(
			model,
			tokenizer,
			eval_loader,
			device,
			args.max_length,
		)

		train_loss = (
			running_loss
			/ max(step_count, 1)
		)

		print(
			f"epoch {epoch + 1}: "
			f"train_loss={train_loss:.6f} "
			f"eval_loss={metrics['loss']:.6f} "
			f"top1={metrics['top1']:.4f} "
			f"mrr={metrics['mrr']:.4f}"
		)

		if metrics["mrr"] > best_mrr:
			best_mrr = metrics["mrr"]

			model.save(output)

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
						"train_loss": train_loss,
						**metrics,
					},
					file,
					ensure_ascii=False,
					indent=2,
				)


if __name__ == "__main__":
	main()
