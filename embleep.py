#!/usr/bin/python3

import argparse, os, sys

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA

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
	arg_parser.add_argument(
		'-pca', '--pca_components', type=int, default=0, help='number of PCA components (default: 0, disabled)')
	# additional settings
	arg_parser.add_argument(
		'-bs', '--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
	arg_parser.add_argument(
		'-rs', '--seed', type=int, help='random seed for probabilistic components (default: None)')
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
	# initialize PCA
	pca_model = None
	if args.pca_components > 0:
		pca_model = PCA(n_components=args.pca_components, random_state=args.seed)
		assert len(target_data) >= pca_model.n_components, \
			f"[Error] Not enough data to perform PCA ({len(target_data)} < {pca_model.n_components})."
		print(f"Using PCA model with {pca_model.n_components} components.")
	# set pooling function for sentence labeling tasks
	pooling_function = load_pooling_function(args.pooling)
	print(f"Using pooling function '{args.pooling}' (sentence classification only).")

	# source labels are embedding dimensions
	source_labels = [f'dim{d}' for d in range(embedding_model.emb_dim)]
	target_labels = target_data.get_label_types()

	# set up output embedding and label stores
	embeddings = np.zeros((len(target_data), embedding_model.emb_dim))
	labels = []

	# iterate over batches
	eidx = 0
	for bidx, (inputs, cur_labels) in enumerate(target_data.get_batches(args.batch_size)):
		# compute embeddings
		cur_embeddings = embedding_model.embed(inputs)  # list of numpy arrays with dim=(emb_dim, )

		# iterate over input sequences in current batch
		for sidx, sequence in enumerate(inputs):
			# case: labels over sequence
			# (i.e. inputs[sidx] = ['t0', 't1', ..., 'tN'], labels[sidx] = ['l0', 'l1', ..., 'lN'])
			if type(cur_labels[sidx]) is list:
				# iterate over all token embeddings in the current sequence
				for tidx in range(len(sequence)):
					tok_embedding = cur_embeddings[sidx][tidx]  # (emb_dim, )
					embeddings[eidx, :] = tok_embedding
					labels.append(cur_labels[sidx][tidx])
					eidx += 1

			# case: one label for entire sequence
			# (i.e. inputs[sidx] = ['t0', 't1', ..., 'tN'], labels[sidx] = 'l')
			else:
				seq_embedding = pooling_function(cur_embeddings[sidx])  # (seq_len, emb_dim) -> (emb_dim,)
				embeddings[eidx, :] = seq_embedding
				eidx += 1
				seq_softmax = softmax(seq_embedding)  # (emb_dim,)
				cur_labels.append(cur_labels[sidx])

		# print progress
		sys.stdout.write(f"\r[{((bidx * args.batch_size)*100)/len(target_data._inputs):.2f}%] Computing embeddings...")
		sys.stdout.flush()
	print("\r", end='')

	print(f"Computed embeddings for {len(target_data)} items.")

	# compute PCA
	if pca_model is not None:
		print(f"Applying PCA to reduce embeddings to {pca_model.n_components} components...")
		embeddings = pca_model.fit_transform(embeddings)
		source_labels = [f'dim{d}' for d in range(pca_model.n_components)]

	# apply softmax to embeddings
	embeddings = softmax(embeddings, axis=1)

	# set up output file
	print(f"Writing LEEP output to '{args.output_path}'...")
	leep_output = LeepWriter(args.output_path)
	# write LEEP header
	leep_output.write_header(source_labels, target_labels)
	# write embeddings and labels
	leep_output.write_instances(embeddings, labels)
	# close LEEP output file pointer
	leep_output.close()
	print(f"Saved LEEP output for {embeddings.shape[0]} instances to '{args.output_path}'.")


if __name__ == '__main__':
	main()
