#!/usr/bin/python3
import argparse
import logging
import os
import sys

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from typing import List, Tuple

# local imports
from project.src.utils.data import LabelledDataset
from project.src.utils.embeddings import load_embeddings, load_pooling_function


def encode_dataset(dataset: LabelledDataset, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    # load embedding model
    embedding_model = load_embeddings(
        args.embedding_model,
        tokenized=(args.task == 'token_classification'),
        static=True
    )
    logging.info(f"Loaded {embedding_model}.")

    # initialize PCA
    pca_model = None
    if args.pca_components > 0:
        pca_model = PCA(n_components=args.pca_components, random_state=args.seed)
        assert len(dataset) >= pca_model.n_components, \
            f"[Error] Not enough data to perform PCA ({len(dataset)} < {pca_model.n_components})."
        logging.info(f"Using PCA model with {pca_model.n_components} components.")
    # set pooling function for sentence labeling tasks
    if args.pooling:
        pooling_function = load_pooling_function(args.pooling)
        logging.info(f"Using pooling function '{args.pooling}' (sentence classification only).")
    else:
        pooling_function = lambda x: x # return identity
        logging.info(f"Using all token-level embeddings (no pooling).")

    # set up output embedding and label stores
    embeddings = np.zeros((len(dataset), embedding_model.emb_dim))
    labels = []

    # iterate over batches
    eidx = 0
    for bidx, (inputs, cur_labels, num_remaining) in enumerate(dataset.get_batches(args.batch_size)):
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

    return np.array(embeddings), np.array(labels)
