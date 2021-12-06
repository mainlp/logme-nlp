import argparse
import logging
from typing import List, Tuple

from datasets import load_dataset


def get_dataset(args: argparse.Namespace) -> Tuple[List[str], List[str], List[str], List[str]]:
    if args.dataset:
        train = load_dataset(str(args.dataset), split="train")
        test = load_dataset(str(args.dataset), split="test")
        logging.debug(f"Dataset Info: {train}")

        try:
            X_train, y_train = train[args.text_column], train[args.label_column]
            X_test, y_test = test[args.text_column], test[args.label_column]
        except (IndexError, KeyError):
            logging.error(f"Cannot find indices for the text or labels. Please try again")
            exit(1)

    elif args.train_path and args.test_path:
        if args.train_path.endswith("csv"):
            train = load_dataset("csv", data_files=str(args.train_path))["train"]
            test = load_dataset("csv", data_files=str(args.test_path))["train"]
        else:
            train = load_dataset("tsv", data_files=str(args.train_path))["train"]
            test = load_dataset("tsv", data_files=str(args.test_path))["train"]
        logging.debug(f"Dataset Info: {train}")

        try:
            X_train, y_train = train[args.text_column], train[args.label_column]
            X_test, y_test = test[args.text_column], test[args.label_column]
        except (IndexError, KeyError):
            logging.error(f"Cannot find indices for the text or labels. Please try again")
            exit(1)
    else:
        logging.error(f"Cannot find dataset or path, please check and try again.")
        exit(1)

    for yidx, label_train in enumerate(y_train):
        if (type(label_train) is str) and (' ' in label_train):
            y_train[yidx] = label_train.split(' ')
    for yidx, label_test in enumerate(y_test):
        if (type(label_test) is str) and (' ' in label_test):
            y_test[yidx] = label_test.split(' ')

    return X_train, y_train, X_test, y_test
