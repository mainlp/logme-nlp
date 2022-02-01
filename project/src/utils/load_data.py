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
		custom_dataset = load_dataset('csv', data_files={
			'train': args.train_path,
			'test': args.test_path
		})
		train = custom_dataset['train']
		test = custom_dataset['test']
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

	# split pre-tokenized data on space
	if args.task == 'token_classification':
		for xidx, text_train in enumerate(X_train):
			X_train[xidx] = text_train.split(' ')
		for xidx, text_test in enumerate(X_test):
			X_test[xidx] = text_test.split(' ')

	for yidx, label_train in enumerate(y_train):
		if args.task == 'token_classification':
			y_train[yidx] = [int(lbl) for lbl in label_train.split(' ')]
	for yidx, label_test in enumerate(y_test):
		if args.task == 'token_classification':
			y_test[yidx] = [int(lbl) for lbl in label_test.split(' ')]

	return X_train, y_train, X_test, y_test
