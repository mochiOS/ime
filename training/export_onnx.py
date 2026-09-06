import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import ImeStudent


class ExportModel(nn.Module):
	def __init__(self, student):
		super().__init__()
		self.student = student

	def forward(
		self,
		input_ids,
		attention_mask,
	):
		return self.student(
			input_ids=input_ids,
			attention_mask=attention_mask,
		).unsqueeze(-1)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--output", required=True)
	parser.add_argument("--max-length", type=int, default=96)
	return parser.parse_args()


def main():
	args = parse_args()

	student = ImeStudent.load(
		args.model,
		torch.device("cpu"),
	)

	student.eval()

	model = ExportModel(student)
	model.eval()

	input_ids = torch.zeros(
		(1, args.max_length),
		dtype=torch.long,
	)

	attention_mask = torch.ones(
		(1, args.max_length),
		dtype=torch.long,
	)

	output = Path(args.output)
	output.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	torch.onnx.export(
		model,
		(
			input_ids,
			attention_mask,
		),
		output,
		input_names=[
			"input_ids",
			"attention_mask",
		],
		output_names=[
			"score",
		],
		dynamic_axes={
			"input_ids": {
				0: "batch",
				1: "sequence",
			},
			"attention_mask": {
				0: "batch",
				1: "sequence",
			},
			"score": {
				0: "batch",
			},
		},
		opset_version=17,
		dynamo=False,
	)

	print(output)


if __name__ == "__main__":
	main()
