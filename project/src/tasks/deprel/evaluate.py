#!/usr/bin/python3

import argparse, os, sys

from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from project.src.utils.data import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Token Classification Evaluation')
	arg_parser.add_argument('tgt_path', help='path to target CSV')
	arg_parser.add_argument('prd_path', help='path to predicted CSV')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	tgt_data = LabelledDataset.from_path(args.tgt_path)
	print(f"Loaded target data {tgt_data} from '{args.tgt_path}'.")
	prd_data = LabelledDataset.from_path(args.prd_path)
	print(f"Loaded predicted data {prd_data} from '{args.prd_path}'.")

	print(list(tgt_data.get_flattened_labels())[:10])
	print(list(prd_data.get_flattened_labels())[:10])

	f1 = f1_score(
		list(tgt_data.get_flattened_labels()),
		list(prd_data.get_flattened_labels()),
		average='micro'
	)
	print(f"F1: {f1 * 100:.2f}%")


if __name__ == '__main__':
	main()
