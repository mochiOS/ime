import argparse

def main():
\tparser = argparse.ArgumentParser()
\tparser.add_argument("--model", required=True)
\tparser.add_argument("--output", required=True)
\targs = parser.parse_args()
\tprint("ONNX export placeholder:", args.model, "->", args.output)

if __name__ == "__main__":
\tmain()
