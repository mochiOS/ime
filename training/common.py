from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, BertConfig, BertModel


ERROR_CLASSES = 5
MAX_LENGTH = 96


class RankingDataset(Dataset):
	def __init__(self, rows):
		self.rows = rows

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, index):
		return self.rows[index]


def collate_groups(groups):
	readings = []
	candidates = []
	group_sizes = []
	labels = []
	error_types = []

	for group in groups:
		items = [
			(group["positive"], 1.0, 0),
			*[
				(
					item["text"],
					0.0,
					int(item["error_type"]),
				)
				for item in group["negatives"]
			],
		]

		group_sizes.append(len(items))

		for text, label, error_type in items:
			readings.append(group["reading"])
			candidates.append(text)
			labels.append(label)
			error_types.append(error_type)

	return {
		"readings": readings,
		"candidates": candidates,
		"group_sizes": group_sizes,
		"labels": torch.tensor(labels, dtype=torch.float32),
		"error_types": torch.tensor(error_types, dtype=torch.long),
	}


def collate_rank_only(groups):
	readings = []
	candidates = []
	group_sizes = []

	for group in groups:
		items = [
			group["positive"],
			*[item["text"] for item in group["negatives"]],
		]

		group_sizes.append(len(items))

		for text in items:
			readings.append(group["reading"])
			candidates.append(text)

	return {
		"readings": readings,
		"candidates": candidates,
		"group_sizes": group_sizes,
	}


def tokenize(tokenizer, batch, device, max_length=MAX_LENGTH):
	inputs = tokenizer(
		batch["readings"],
		batch["candidates"],
		padding=True,
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)

	return {
		key: value.to(device)
		for key, value in inputs.items()
	}


def split_scores(scores, group_sizes):
	result = []
	offset = 0

	for size in group_sizes:
		result.append(scores[offset:offset + size])
		offset += size

	return result


def listwise_loss(scores, group_sizes):
	losses = []

	for group in split_scores(scores, group_sizes):
		if group.numel() == 0:
			continue

		losses.append(
			-F.log_softmax(group.float(), dim=0)[0]
		)

	if not losses:
		return scores.new_tensor(0.0)

	return torch.stack(losses).mean()


def pairwise_margin_loss(scores, group_sizes, margin=1.0):
	losses = []

	for group in split_scores(scores, group_sizes):
		if group.numel() <= 1:
			continue

		positive = group[0].float()
		negatives = group[1:].float()

		losses.append(
			F.relu(margin - positive + negatives).mean()
		)

	if not losses:
		return scores.new_tensor(0.0)

	return torch.stack(losses).mean()


def distillation_loss(
	student_scores,
	teacher_scores,
	group_sizes,
	temperature=2.0,
):
	losses = []

	for student, teacher in zip(
		split_scores(student_scores, group_sizes),
		split_scores(teacher_scores, group_sizes),
	):
		student_log_prob = F.log_softmax(
			student.float() / temperature,
			dim=0,
		)

		teacher_prob = F.softmax(
			teacher.float() / temperature,
			dim=0,
		)

		losses.append(
			F.kl_div(
				student_log_prob,
				teacher_prob,
				reduction="sum",
			)
			* temperature
			* temperature
		)

	if not losses:
		return student_scores.new_tensor(0.0)

	return torch.stack(losses).mean()


def ranking_metrics(scores, group_sizes):
	total = 0
	top1 = 0
	mrr = 0.0

	for group in split_scores(scores, group_sizes):
		order = torch.argsort(
			group.float(),
			descending=True,
		)

		position = (
			order == 0
		).nonzero(as_tuple=False)[0, 0].item()

		top1 += int(position == 0)
		mrr += 1.0 / (position + 1)
		total += 1

	if total == 0:
		return {
			"top1": 0.0,
			"mrr": 0.0,
		}

	return {
		"top1": top1 / total,
		"mrr": mrr / total,
	}


class ImeTeacher(nn.Module):
	def __init__(self, model_name=None, model_path=None):
		super().__init__()

		if model_path is not None:
			self.encoder = AutoModel.from_pretrained(model_path)
		elif model_name is not None:
			self.encoder = AutoModel.from_pretrained(model_name)
		else:
			raise ValueError("model_name or model_path is required")

		hidden = self.encoder.config.hidden_size

		self.rank_head = nn.Linear(hidden, 1)
		self.correctness_head = nn.Linear(hidden, 1)
		self.error_head = nn.Linear(hidden, ERROR_CLASSES)

	def forward(self, **inputs):
		output = self.encoder(**inputs)

		hidden = output.last_hidden_state[:, 0]

		rank_hidden = hidden.to(
			dtype=self.rank_head.weight.dtype
		)

		correctness_hidden = hidden.to(
			dtype=self.correctness_head.weight.dtype
		)

		error_hidden = hidden.to(
			dtype=self.error_head.weight.dtype
		)

		return {
			"rank": self.rank_head(
				rank_hidden
			).squeeze(-1),

			"correctness": self.correctness_head(
				correctness_hidden
			).squeeze(-1),

			"error": self.error_head(
				error_hidden
			),
		}

	def save(self, output):
		output = Path(output)
		output.mkdir(parents=True, exist_ok=True)

		self.encoder.save_pretrained(output)

		torch.save(
			{
				"rank_head": self.rank_head.state_dict(),
				"correctness_head": self.correctness_head.state_dict(),
				"error_head": self.error_head.state_dict(),
			},
			output / "heads.pt",
		)

	@classmethod
	def load(cls, path, device):
		model = cls(model_path=path)

		heads = torch.load(
			Path(path) / "heads.pt",
			map_location="cpu",
			weights_only=True,
		)

		model.rank_head.load_state_dict(heads["rank_head"])
		model.correctness_head.load_state_dict(
			heads["correctness_head"]
		)
		model.error_head.load_state_dict(heads["error_head"])

		return model.to(device)


class ImeStudent(nn.Module):
	def __init__(self, encoder):
		super().__init__()

		self.encoder = encoder

		self.rank_head = nn.Linear(
			encoder.config.hidden_size,
			1,
		)

	def forward(self, **inputs):
		output = self.encoder(**inputs)

		return self.rank_head(
			output.last_hidden_state[:, 0]
		).squeeze(-1)

	def save(self, output):
		output = Path(output)
		output.mkdir(parents=True, exist_ok=True)

		self.encoder.save_pretrained(output)

		torch.save(
			self.rank_head.state_dict(),
			output / "rank_head.pt",
		)

	@classmethod
	def create(
		cls,
		vocab_size,
		max_length=MAX_LENGTH,
		hidden_size=256,
		layers=4,
		heads=4,
		intermediate_size=768,
	):
		config = BertConfig(
			vocab_size=vocab_size,
			hidden_size=hidden_size,
			num_hidden_layers=layers,
			num_attention_heads=heads,
			intermediate_size=intermediate_size,
			max_position_embeddings=max(128, max_length),
			type_vocab_size=2,
			hidden_dropout_prob=0.1,
			attention_probs_dropout_prob=0.1,
		)

		return cls(BertModel(config))

	@classmethod
	def load(cls, path, device):
		encoder = AutoModel.from_pretrained(path)
		model = cls(encoder)

		state = torch.load(
			Path(path) / "rank_head.pt",
			map_location="cpu",
			weights_only=True,
		)

		model.rank_head.load_state_dict(state)

		return model.to(device)
