#!/usr/bin/python3

import argparse, os, sys

from scipy.special import softmax

# local imports
from utils.datasets import LabelledDataset, LeepWriter
from utils.embeddings import load_embeddings, load_pooling_function


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Embedding to LEEP')
	# data setup
	arg_parser.add_argument('target_path', help='path to target data')
	arg_parser.add_argument('output_path', help='path to output file')
	# embedding model setup
	arg_parser.add_argument('embedding_model', help='embedding model identifier')
	arg_parser.add_argument(
		'-pl', '--pooling', default='mean', help='pooling strategy for sentence classification (default: mean)')
	# additional settings
	arg_parser.add_argument(
		'-bs', '--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# check if output file exists
	if os.path.exists(args.output_path):
		response = None
		while response not in ['y', 'n']:
			response = input(f"File '{args.output_path}' already exists. Overwrite? [y/n] ")
		if response == 'n':
			exit(1)

	# load target dataset
	target_data = LabelledDataset.from_path(args.target_path)
	print(f"Loaded {target_data}.")

	# load embedding model
	embedding_model = load_embeddings(args.embedding_model, static=True)
	print(f"Loaded {embedding_model}.")
	pooling_function = load_pooling_function(args.pooling)
	print(f"Using pooling function '{args.pooling}' (sentence classification only).")

	# set up output file
	leep_output = LeepWriter(args.output_path)
	# source labels are embedding dimensions
	source_labels = [f'dim{d}' for d in range(embedding_model.emb_dim)]
	target_labels = target_data.get_label_types()
	# write LEEP header
	leep_output.write_header(source_labels, target_labels)
	print(f"Writing LEEP output to '{args.output_path}'.")

	# iterate over batches
	for bidx, (inputs, labels) in enumerate(target_data.get_batches(args.batch_size)):
		# compute embeddings
		embeddings = embedding_model.embed(inputs)  # list of numpy arrays with dim=(emb_dim, )

		# iterate over input sequences in current batch
		for sidx, sequence in enumerate(inputs):
			# case: labels over sequence
			# (i.e. inputs[sidx] = ['t0', 't1', ..., 'tN'], labels[sidx] = ['l0', 'l1', ..., 'lN'])
			if type(labels[sidx]) is list:
				# iterate over all token embeddings in the current sequence
				for tidx in range(len(sequence)):
					tok_embedding = embeddings[sidx][tidx]  # (emb_dim, )
					tok_softmax = softmax(tok_embedding)  # (emb_dim, )
					tok_label = labels[sidx][tidx]
					# write to output file
					leep_output.write_instance(tok_softmax, tok_label)

			# case: one label for entire sequence
			# (i.e. inputs[sidx] = ['t0', 't1', ..., 'tN'], labels[sidx] = 'l')
			else:
				seq_embedding = pooling_function(embeddings[sidx])  # (seq_len, emb_dim) -> (emb_dim,)
				seq_softmax = softmax(seq_embedding)  # (emb_dim,)
				seq_label = labels[sidx]
				# write to output file
				leep_output.write_instance(seq_softmax, seq_label)

		# print progress
		sys.stdout.write(f"\r[{((bidx * args.batch_size)*100)/len(target_data._inputs):.2f}%] Computing embeddings...")
		sys.stdout.flush()
	print("\r", end='')

	# close LEEP output file pointer
	leep_output.close()

	print(f"Computed embeddings for {len(target_data)} items.")


if __name__ == '__main__':
	main()
