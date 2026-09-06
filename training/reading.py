import sys

from sudachipy import Dictionary


def katakana_to_hiragana(text: str) -> str:
	return "".join(
		chr(ord(ch) - 0x60)
		if "\u30a1" <= ch <= "\u30f6"
		else ch
		for ch in text
	)


def main() -> None:
	tokenizer = Dictionary().create()

	for line in sys.stdin:
		line = line.rstrip("\r\n")

		if not line:
			print()
			continue

		morphemes = tokenizer.tokenize(line)

		reading = "".join(
			morpheme.reading_form()
			for morpheme in morphemes
		)

		print(katakana_to_hiragana(reading))


if __name__ == "__main__":
	main()

# さっきPythonにあらがうとか言ってたよな、あれは嘘だ!!無理だ！！！
#
#               - tas0dev