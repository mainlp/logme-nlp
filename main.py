#!/usr/bin/python3

import argparse
import logging
import sys

# from dotenv import load_dotenv
import numpy as np

# from project.src.preprocessing.tokenize import tokenize_text
from project.src.utils.data import LabelledDataset
from project.src.utils.encode_data import encode_dataset
from project.src.utils.leep import LogExpectedEmpiricalPrediction
from project.src.utils.logme import LogME
from project.src.utils.load_data import get_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# load_dotenv(verbose=True)


def main(args: argparse.Namespace):
    # load dataset from HF or custom
    X_train, y_train, _, _ = get_dataset(args)

    # tokenize text
    # logging.info("Tokenizing text...")
    # tokenized_text = tokenize_text(args, X_train)

    # create LabelledDataset object
    dataset = LabelledDataset(inputs=X_train, labels=y_train)
    logging.info(f"Loaded {dataset}.")

    # encode dataset
    embeddings, labels = encode_dataset(dataset, args)
    # leep_filepath = encode_dataset(dataset, args)

    # leep = LogExpectedEmpiricalPrediction(leep_filepath)
    # logging.info(f"LEEP: {leep.compute_leep()}")

    logme = LogME(regression=False)
    score = logme.fit(embeddings, labels)
    logging.info(f"LogME: {score}")
    with open(f"results_{args.dataset}.txt", "a") as f:
        f.write(f"{args.embedding_model} | {args.dataset} | LogME: {score}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framework for LEEP')

    parser.add_argument('--dataset', type=str, nargs='?', help='Dataset from the HuggingFace Dataset library.')
    parser.add_argument('--train_path', type=str, nargs='?', help='Path to the training set.')
    parser.add_argument('--test_path', type=str, nargs='?', help='Path to the test set.')

    parser.add_argument('--text_column', type=str, nargs='?', help='Indicate which column to use for features.')
    parser.add_argument('--label_column', type=str, nargs='?', help='Indicate which column to use for gold labels.')

    parser.add_argument('--tokenizer', type=str, nargs='?', help='Indicate which tokenizer to use for text.')
    parser.add_argument('--output_file', type=str, nargs='?', help='The name of the output file.')

    parser.add_argument('--embedding_model', type=str, nargs='?', help='embedding model identifier')
    parser.add_argument('--pooling', default='mean',
                        help='pooling strategy for sentence classification (default: mean)')
    parser.add_argument('--pca_components', type=int, default=0, help='number of PCA components (default: 0, disabled)')
    # additional settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='maximum number of sentences per batch (default: 64)')
    parser.add_argument('--seed', type=int, help='random seed for probabilistic components (default: None)')

    main(parser.parse_args())
