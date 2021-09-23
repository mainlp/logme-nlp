#!/usr/bin/python3
import argparse
import logging
import os
import sys

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA

# local imports
from project.src.utils.leep_data import LabelledDataset, LeepWriter
from project.src.utils.embeddings import load_embeddings, load_pooling_function


def encode_dataset(dataset: LabelledDataset, args: argparse.Namespace) -> str:
    # load embedding model
    embedding_model = load_embeddings(args.embedding_model, static=True)
    logging.info(f"Loaded {embedding_model}.")

    # initialize PCA
    pca_model = None
    if args.pca_components > 0:
        pca_model = PCA(n_components=args.pca_components, random_state=args.seed)
        assert len(dataset) >= pca_model.n_components, \
            f"[Error] Not enough data to perform PCA ({len(dataset)} < {pca_model.n_components})."
        logging.info(f"Using PCA model with {pca_model.n_components} components.")
    # set pooling function for sentence labeling tasks
    pooling_function = load_pooling_function(args.pooling)
    logging.info(f"Using pooling function '{args.pooling}' (sentence classification only).")

    # source labels are embedding dimensions
    source_labels = [f'dim{d}' for d in range(embedding_model.emb_dim)]
    target_labels = dataset.get_label_types()

    # set up output embedding and label stores
    embeddings = np.zeros((len(dataset), embedding_model.emb_dim))
    labels = []

    # iterate over batches
    eidx = 0
    for bidx, (inputs, cur_labels) in enumerate(dataset.get_batches(args.batch_size)):
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
                labels.append(cur_labels[sidx])
                eidx += 1

        # print progress
        sys.stdout.write(
                f"\r[{((bidx * args.batch_size) * 100) / len(dataset._inputs):.2f}%] Computing embeddings...")
        sys.stdout.flush()
    print("\r", end='')

    logging.info(f"Computed embeddings for {len(dataset)} items.")

    # compute PCA
    if pca_model is not None:
        logging.info(f"Applying PCA to reduce embeddings to {pca_model.n_components} components...")
        embeddings = pca_model.fit_transform(embeddings)
        source_labels = [f'dim{d}' for d in range(pca_model.n_components)]

    # apply softmax to embeddings
    embeddings = softmax(embeddings, axis=1)

    # set up output file
    leep_file = os.path.join(os.getenv('OUTPUT_PATH'), args.output_file)

    logging.info(f"Writing LEEP output to '{leep_file}'...")
    leep_output = LeepWriter(leep_file)
    # write LEEP header
    leep_output.write_header(source_labels, target_labels)
    # write embeddings and labels
    leep_output.write_instances(embeddings, labels)
    # close LEEP output file pointer
    leep_output.close()
    logging.info(f"Saved LEEP output for {embeddings.shape[0]} instances to '{leep_file}'.")

    return leep_file
