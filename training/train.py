import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BertConfig,
	BertForSequenceClassification,
	get_linear_schedule_with_warmup,
)


DEFAULT_TEACHER = "ku-nlp/deberta-v3-base-japanese"

DEFAULT_MAX_LENGTH = 96
DEFAULT_SEED = 42
DEFAULT_EVAL_RATIO = 0.1

DEFAULT_TEACHER_EPOCHS = 5
DEFAULT_STUDENT_EPOCHS = 12

DEFAULT_TEACHER_BATCH_SIZE = 2
DEFAULT_STUDENT_BATCH_SIZE = 8

DEFAULT_TEACHER_LEARNING_RATE = 2e-5
DEFAULT_STUDENT_LEARNING_RATE = 5e-5

DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.08

DEFAULT_GRADIENT_ACCUMULATION = 8

DEFAULT_DISTILL_TEMPERATURE = 2.0
DEFAULT_DISTILL_WEIGHT = 0.7

STUDENT_HIDDEN_SIZE = 256
STUDENT_LAYERS = 4
STUDENT_HEADS = 4
STUDENT_INTERMEDIATE_SIZE = 768


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--input",
		required=True,
	)

	parser.add_argument(
		"--output",
		default="training/model",
	)

	parser.add_argument(
		"--teacher-model",
		default=DEFAULT_TEACHER,
	)

	parser.add_argument(
		"--max-length",
		type=int,
		default=DEFAULT_MAX_LENGTH,
	)

	parser.add_argument(
		"--eval-ratio",
		type=float,
		default=DEFAULT_EVAL_RATIO,
	)

	parser.add_argument(
		"--teacher-epochs",
		type=int,
		default=DEFAULT_TEACHER_EPOCHS,
	)

	parser.add_argument(
		"--student-epochs",
		type=int,
		default=DEFAULT_STUDENT_EPOCHS,
	)

	parser.add_argument(
		"--teacher-batch-size",
		type=int,
		default=DEFAULT_TEACHER_BATCH_SIZE,
	)

	parser.add_argument(
		"--student-batch-size",
		type=int,
		default=DEFAULT_STUDENT_BATCH_SIZE,
	)

	parser.add_argument(
		"--teacher-learning-rate",
		type=float,
		default=DEFAULT_TEACHER_LEARNING_RATE,
	)

	parser.add_argument(
		"--student-learning-rate",
		type=float,
		default=DEFAULT_STUDENT_LEARNING_RATE,
	)

	parser.add_argument(
		"--gradient-accumulation",
		type=int,
		default=DEFAULT_GRADIENT_ACCUMULATION,
	)

	parser.add_argument(
		"--temperature",
		type=float,
		default=DEFAULT_DISTILL_TEMPERATURE,
	)

	parser.add_argument(
		"--distill-weight",
		type=float,
		default=DEFAULT_DISTILL_WEIGHT,
	)

	parser.add_argument(
		"--seed",
		type=int,
		default=DEFAULT_SEED,
	)

	return parser.parse_args()


def unescape_field(value: str) -> str:
	result = []
	index = 0

	while index < len(value):
		if value[index] != "\\":
			result.append(value[index])
			index += 1
			continue

		if index + 1 >= len(value):
			result.append("\\")
			index += 1
			continue

		next_char = value[index + 1]

		if next_char == "t":
			result.append("\t")
		elif next_char == "r":
			result.append("\r")
		elif next_char == "n":
			result.append("\n")
		elif next_char == "\\":
			result.append("\\")
		else:
			result.append("\\")
			result.append(next_char)

		index += 2

	return "".join(result)


def load_groups(path: Path) -> list[dict]:
	raw_groups = defaultdict(
		lambda: {
			"positive": None,
			"negatives": [],
		}
	)

	with path.open(
		"r",
		encoding="utf-8",
	) as file:
		for line_number, line in enumerate(file, start=1):
			line = line.rstrip("\r\n")

			if not line:
				continue

			fields = line.split("\t")

			if len(fields) != 3:
				raise ValueError(
					f"{path}:{line_number}: "
					f"expected 3 fields"
				)

			label_text, reading, candidate = fields

			label = int(label_text)

			if label not in (0, 1):
				raise ValueError(
					f"{path}:{line_number}: "
					f"invalid label {label}"
				)

			reading = unescape_field(reading)
			candidate = unescape_field(candidate)

			group = raw_groups[reading]

			if label == 1:
				if group["positive"] is not None:
					if group["positive"] != candidate:
						raise ValueError(
							f"{path}:{line_number}: "
							"multiple positive candidates "
							f"for reading {reading!r}"
						)

				group["positive"] = candidate
			else:
				group["negatives"].append(candidate)

	groups = []

	for reading, group in raw_groups.items():
		positive = group["positive"]

		if positive is None:
			continue

		seen = {positive}

		negatives = []

		for candidate in group["negatives"]:
			if candidate in seen:
				continue

			seen.add(candidate)
			negatives.append(candidate)

		if not negatives:
			continue

		groups.append(
			{
				"reading": reading,
				"positive": positive,
				"negatives": negatives,
			}
		)

	if len(groups) < 2:
		raise ValueError(
			"dataset must contain at least two usable readings"
		)

	return groups


def split_groups(
	groups: list[dict],
	eval_ratio: float,
	seed: int,
) -> tuple[list[dict], list[dict]]:
	if not 0.0 < eval_ratio < 1.0:
		raise ValueError(
			"--eval-ratio must be between 0 and 1"
		)

	groups = list(groups)

	random.Random(seed).shuffle(groups)

	eval_count = max(
		1,
		int(len(groups) * eval_ratio),
	)

	eval_count = min(
		eval_count,
		len(groups) - 1,
	)

	eval_groups = groups[:eval_count]
	train_groups = groups[eval_count:]

	return train_groups, eval_groups


class GroupDataset(Dataset):
	def __init__(
		self,
		groups: list[dict],
	):
		self.groups = groups

	def __len__(self) -> int:
		return len(self.groups)

	def __getitem__(self, index: int) -> dict:
		return self.groups[index]


def collate_groups(groups: list[dict]) -> dict:
	readings = []
	candidates = []
	group_sizes = []

	for group in groups:
		group_candidates = [
			group["positive"],
			*group["negatives"],
		]

		group_sizes.append(
			len(group_candidates)
		)

		for candidate in group_candidates:
			readings.append(
				group["reading"]
			)

			candidates.append(
				candidate
			)

	return {
		"readings": readings,
		"candidates": candidates,
		"group_sizes": group_sizes,
	}


def move_inputs(
	inputs: dict,
	device: torch.device,
) -> dict:
	return {
		key: value.to(device)
		for key, value in inputs.items()
	}


def tokenize_batch(
	tokenizer,
	batch: dict,
	max_length: int,
	device: torch.device,
) -> dict:
	inputs = tokenizer(
		batch["readings"],
		batch["candidates"],
		padding=True,
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)

	return move_inputs(
		inputs,
		device,
	)


def split_scores(
	scores: torch.Tensor,
	group_sizes: list[int],
) -> list[torch.Tensor]:
	result = []

	offset = 0

	for size in group_sizes:
		result.append(
			scores[offset:offset + size]
		)

		offset += size

	return result


def listwise_loss(
	scores: torch.Tensor,
	group_sizes: list[int],
) -> torch.Tensor:
	group_scores = split_scores(
		scores,
		group_sizes,
	)

	losses = []

	for values in group_scores:
		log_probabilities = F.log_softmax(
			values,
			dim=0,
		)

		losses.append(
			-log_probabilities[0]
		)

	return torch.stack(losses).mean()


def distillation_loss(
	student_scores: torch.Tensor,
	teacher_scores: torch.Tensor,
	group_sizes: list[int],
	temperature: float,
) -> torch.Tensor:
	student_groups = split_scores(
		student_scores,
		group_sizes,
	)

	teacher_groups = split_scores(
		teacher_scores,
		group_sizes,
	)

	losses = []

	for student, teacher in zip(
		student_groups,
		teacher_groups,
	):
		student_log_probabilities = F.log_softmax(
			student / temperature,
			dim=0,
		)

		teacher_probabilities = F.softmax(
			teacher / temperature,
			dim=0,
		)

		loss = F.kl_div(
			student_log_probabilities,
			teacher_probabilities,
			reduction="sum",
		)

		losses.append(
			loss * temperature * temperature
		)

	return torch.stack(losses).mean()


def evaluate(
	model,
	tokenizer,
	loader: DataLoader,
	device: torch.device,
	max_length: int,
) -> tuple[float, float, float]:
	model.eval()

	total_groups = 0
	top1_correct = 0
	reciprocal_rank_sum = 0.0
	loss_sum = 0.0

	with torch.inference_mode():
		for batch in loader:
			inputs = tokenize_batch(
				tokenizer,
				batch,
				max_length,
				device,
			)

			outputs = model(**inputs)

			scores = outputs.logits
			.squeeze(-1)
			.float()

			loss = listwise_loss(
				scores,
				batch["group_sizes"],
			)

			loss_sum += loss.item()

			for group_scores in split_scores(
				scores,
				batch["group_sizes"],
			):
				order = torch.argsort(
					group_scores,
					descending=True,
				)

				position = (
					order == 0
				).nonzero(
					as_tuple=False
				)[0, 0].item()

				if position == 0:
					top1_correct += 1

				reciprocal_rank_sum += (
					1.0 / (position + 1)
				)

				total_groups += 1

	model.train()

	if total_groups == 0:
		return 0.0, 0.0, 0.0

	return (
		loss_sum / max(len(loader), 1),
		top1_correct / total_groups,
		reciprocal_rank_sum / total_groups,
	)


def create_optimizer(
	model,
	learning_rate: float,
) -> AdamW:
	return AdamW(
		model.parameters(),
		lr=learning_rate,
		weight_decay=DEFAULT_WEIGHT_DECAY,
	)


def create_scheduler(
	optimizer: AdamW,
	loader_length: int,
	epochs: int,
	gradient_accumulation: int,
):
	steps_per_epoch = math.ceil(
		loader_length
		/ gradient_accumulation
	)

	total_steps = (
		steps_per_epoch
		* epochs
	)

	warmup_steps = int(
		total_steps
		* DEFAULT_WARMUP_RATIO
	)

	return get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_steps,
	)


def train_teacher(
	model,
	tokenizer,
	train_loader: DataLoader,
	eval_loader: DataLoader,
	device: torch.device,
	args: argparse.Namespace,
	output_dir: Path,
) -> None:
	optimizer = create_optimizer(
		model,
		args.teacher_learning_rate,
	)

	scheduler = create_scheduler(
		optimizer,
		len(train_loader),
		args.teacher_epochs,
		args.gradient_accumulation,
	)

	scaler = torch.amp.GradScaler(
		"cuda",
		enabled=device.type == "cuda",
	)

	best_mrr = -1.0

	optimizer.zero_grad(
		set_to_none=True,
	)

	for epoch in range(args.teacher_epochs):
		model.train()

		running_loss = 0.0

		for step, batch in enumerate(
			train_loader,
			start=1,
		):
			inputs = tokenize_batch(
				tokenizer,
				batch,
				args.max_length,
				device,
			)

			with torch.autocast(
				device_type=device.type,
				dtype=torch.bfloat16
				if device.type == "cuda"
				and torch.cuda.is_bf16_supported()
				else torch.float16,
				enabled=device.type == "cuda",
			):
				outputs = model(**inputs)

				scores = outputs.logits.squeeze(-1)

				loss = listwise_loss(
					scores,
					batch["group_sizes"],
				)

				loss = (
					loss
					/ args.gradient_accumulation
				)

			scaler.scale(loss).backward()

			running_loss += (
				loss.item()
				* args.gradient_accumulation
			)

			if (
				step % args.gradient_accumulation == 0
				or step == len(train_loader)
			):
				scaler.unscale_(optimizer)

				torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					1.0,
				)

				scaler.step(optimizer)
				scaler.update()

				optimizer.zero_grad(
					set_to_none=True,
				)

				scheduler.step()

		eval_loss, top1, mrr = evaluate(
			model,
			tokenizer,
			eval_loader,
			device,
			args.max_length,
		)

		print(
			f"[teacher] epoch={epoch + 1} "
			f"train_loss={running_loss / max(len(train_loader), 1):.6f} "
			f"eval_loss={eval_loss:.6f} "
			f"top1={top1:.4f} "
			f"mrr={mrr:.4f}"
		)

		if mrr > best_mrr:
			best_mrr = mrr

			output_dir.mkdir(
				parents=True,
				exist_ok=True,
			)

			model.save_pretrained(
				output_dir
			)

			tokenizer.save_pretrained(
				output_dir
			)


def create_student(
	tokenizer,
	max_length: int,
) -> BertForSequenceClassification:
	config = BertConfig(
		vocab_size=len(tokenizer),
		hidden_size=STUDENT_HIDDEN_SIZE,
		num_hidden_layers=STUDENT_LAYERS,
		num_attention_heads=STUDENT_HEADS,
		intermediate_size=STUDENT_INTERMEDIATE_SIZE,
		max_position_embeddings=max(
			128,
			max_length,
		),
		type_vocab_size=2,
		num_labels=1,
		hidden_dropout_prob=0.1,
		attention_probs_dropout_prob=0.1,
	)

	return BertForSequenceClassification(
		config
	)


def train_student(
	student,
	teacher,
	tokenizer,
	train_loader: DataLoader,
	eval_loader: DataLoader,
	device: torch.device,
	args: argparse.Namespace,
	output_dir: Path,
) -> None:
	teacher.eval()

	for parameter in teacher.parameters():
		parameter.requires_grad_(False)

	optimizer = create_optimizer(
		student,
		args.student_learning_rate,
	)

	scheduler = create_scheduler(
		optimizer,
		len(train_loader),
		args.student_epochs,
		args.gradient_accumulation,
	)

	scaler = torch.amp.GradScaler(
		"cuda",
		enabled=device.type == "cuda",
	)

	best_mrr = -1.0

	optimizer.zero_grad(
		set_to_none=True,
	)

	for epoch in range(args.student_epochs):
		student.train()

		running_loss = 0.0

		for step, batch in enumerate(
			train_loader,
			start=1,
		):
			inputs = tokenize_batch(
				tokenizer,
				batch,
				args.max_length,
				device,
			)

			with torch.inference_mode():
				teacher_scores = (
					teacher(**inputs)
					.logits
					.squeeze(-1)
					.float()
				)

			with torch.autocast(
				device_type=device.type,
				dtype=torch.bfloat16
				if device.type == "cuda"
				and torch.cuda.is_bf16_supported()
				else torch.float16,
				enabled=device.type == "cuda",
			):
				student_scores = (
					student(**inputs)
					.logits
					.squeeze(-1)
				)

				hard_loss = listwise_loss(
					student_scores,
					batch["group_sizes"],
				)

				soft_loss = distillation_loss(
					student_scores.float(),
					teacher_scores,
					batch["group_sizes"],
					args.temperature,
				)

				loss = (
					(1.0 - args.distill_weight)
					* hard_loss
					+ args.distill_weight
					* soft_loss
				)

				loss = (
					loss
					/ args.gradient_accumulation
				)

			scaler.scale(loss).backward()

			running_loss += (
				loss.item()
				* args.gradient_accumulation
			)

			if (
				step % args.gradient_accumulation == 0
				or step == len(train_loader)
			):
				scaler.unscale_(optimizer)

				torch.nn.utils.clip_grad_norm_(
					student.parameters(),
					1.0,
				)

				scaler.step(optimizer)
				scaler.update()

				optimizer.zero_grad(
					set_to_none=True,
				)

				scheduler.step()

		eval_loss, top1, mrr = evaluate(
			student,
			tokenizer,
			eval_loader,
			device,
			args.max_length,
		)

		print(
			f"[student] epoch={epoch + 1} "
			f"train_loss={running_loss / max(len(train_loader), 1):.6f} "
			f"eval_loss={eval_loss:.6f} "
			f"top1={top1:.4f} "
			f"mrr={mrr:.4f}"
		)

		if mrr > best_mrr:
			best_mrr = mrr

			output_dir.mkdir(
				parents=True,
				exist_ok=True,
			)

			student.save_pretrained(
				output_dir
			)

			tokenizer.save_pretrained(
				output_dir
			)


def main() -> None:
	args = parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(
			args.seed
		)

	device = torch.device(
		"cuda"
		if torch.cuda.is_available()
		else "cpu"
	)

	print(f"device: {device}")

	if device.type == "cuda":
		print(
			f"gpu: {torch.cuda.get_device_name(0)}"
		)

	groups = load_groups(
		Path(args.input)
	)

	train_groups, eval_groups = split_groups(
		groups,
		args.eval_ratio,
		args.seed,
	)

	print(
		f"train readings: {len(train_groups)}"
	)

	print(
		f"eval readings: {len(eval_groups)}"
	)

	tokenizer = AutoTokenizer.from_pretrained(
		args.teacher_model,
		use_fast=True,
	)

	train_dataset = GroupDataset(
		train_groups
	)

	eval_dataset = GroupDataset(
		eval_groups
	)

	teacher_train_loader = DataLoader(
		train_dataset,
		batch_size=args.teacher_batch_size,
		shuffle=True,
		collate_fn=collate_groups,
	)

	student_train_loader = DataLoader(
		train_dataset,
		batch_size=args.student_batch_size,
		shuffle=True,
		collate_fn=collate_groups,
	)

	eval_loader = DataLoader(
		eval_dataset,
		batch_size=args.student_batch_size,
		shuffle=False,
		collate_fn=collate_groups,
	)

	output_root = Path(
		args.output
	)

	teacher_output = (
		output_root / "teacher"
	)

	student_output = (
		output_root / "student"
	)

	print(
		f"loading teacher: {args.teacher_model}"
	)

	teacher = (
		AutoModelForSequenceClassification
		.from_pretrained(
			args.teacher_model,
			num_labels=1,
			ignore_mismatched_sizes=True,
		)
		.to(device)
	)

	train_teacher(
		teacher,
		tokenizer,
		teacher_train_loader,
		eval_loader,
		device,
		args,
		teacher_output,
	)

	del teacher

	if device.type == "cuda":
		torch.cuda.empty_cache()

	print("loading best teacher")

	teacher = (
		AutoModelForSequenceClassification
		.from_pretrained(
			teacher_output,
		)
		.to(device)
	)

	print("creating student")

	student = create_student(
		tokenizer,
		args.max_length,
	).to(device)

	print(
		"student architecture: "
		f"{STUDENT_LAYERS} layers, "
		f"hidden={STUDENT_HIDDEN_SIZE}, "
		f"heads={STUDENT_HEADS}"
	)

	print(
		f"student parameters: "
		f"{sum(parameter.numel() for parameter in student.parameters()):,}"
	)

	train_student(
		student,
		teacher,
		tokenizer,
		student_train_loader,
		eval_loader,
		device,
		args,
		student_output,
	)

	print(
		f"student saved to {student_output}"
	)


if __name__ == "__main__":
	main()