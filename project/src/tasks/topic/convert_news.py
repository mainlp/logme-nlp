#!/usr/bin/python3

import argparse
import csv
import os
import sys

import numpy as np
import transformers

sys.path.append("/../../logme-frame")
from project.src.utils.load_data import get_dataset


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='AGNews - Dataset Conversion')
    arg_parser.add_argument('dataset', help='path to AGNews CSV')
    arg_parser.add_argument('output_path', help='output prefix for corpus in HuggingFace Datasets CSV format')
    arg_parser.add_argument('--text_column', type=str, nargs='?', help='Indicate which column to use for features.')
    arg_parser.add_argument('--label_column', type=str, nargs='?', help='Indicate which column to use for gold labels.')
    arg_parser.add_argument('-t', '--tokenizer', help='name of HuggingFace tokenizer')
    arg_parser.add_argument('-p', '--proportions', default='.7,.1,.2',
                            help='train, dev, test proportions (default: ".7,.1,.2")')
    arg_parser.add_argument('-rs', '--random_seed', type=int, help='seed for probabilistic components (default: None)')
    return arg_parser.parse_args()


def tokenize(text, tokenizer):
    # run raw input trough tokenizer
    tokenization = tokenizer(
            text,
            padding=True, truncation=True, return_special_tokens_mask=True, return_offsets_mapping=True
            )

    # reduce sub-words into words
    tokens = []
    # get string subtokens and offsets
    subtokens = tokenizer.convert_ids_to_tokens(tokenization['input_ids'])
    offsets = tokenization['offset_mapping']
    for stidx, subtoken in enumerate(subtokens):
        # skip special tokens (e.g. [CLS], [SEQ], [PAD])
        if tokenization['special_tokens_mask'][stidx] == 1:
            continue

        # subword re-combination (RoBERTa, BPE)
        if isinstance(tokenizer, transformers.models.gpt2.GPT2TokenizerFast):
            # if current subtoken starts with space-byte, treat as new token
            if subtoken.startswith('Ġ'):
                tokens.append(subtoken[1:])
            # otherwise, append to last token
            else:
                tokens[-1] += subtoken
        # subword re-combination (Other, WordPiece)
        else:
            # if current subtoken is part of the previous token, append to it
            if subtoken.startswith('##'):
                tokens[-1] += subtoken[2:]
            # otherwise, treat as new token
            else:
                tokens.append(subtoken)

    return tokens


def main():
    args = parse_arguments()

    # dataset-specific variables
    # lbl_idx_map = {'World': 0, 'Sports': '1', 'Business': 2, 'Sci/Tech': 3}

    # set random seed
    np.random.seed(args.random_seed)

    # initialize output lines
    output = []

    # load tokenizer
    if args.tokenizer is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, add_prefix_space=True)
        print(f"Loaded tokenizer '{args.tokenizer}' ({tokenizer.__class__.__name__}).")

    # iterate over data points
    X_train, y_train, X_test, y_test = get_dataset(args)
    sentences = X_train + X_test
    labels = y_train + y_test

    for text, label in zip(sentences, labels):
        # tokenize text
        if args.tokenizer is not None:
            text_tokenized = tokenize(text, tokenizer)
            output.append([" ".join(text_tokenized), label])
        # keep text unchanged
        else:
            output.append([text, label])

    # split data into train, dev, test
    proportions = [float(pstr) for pstr in args.proportions.split(',')]
    # check split proportions
    assert sum(proportions) == 1, f"[Error] Split proportions {proportions} do not sum up to 1."
    assert len(proportions) == 3, f"[Error] There must be three proportions (i.e. train, dev, test)."
    # shuffle the data in-place
    np.random.shuffle(output)
    # create splits
    splits = {}
    cursor = 0
    for spidx, split in enumerate(['train', 'dev', 'test']):
        cursor_end = cursor + (round(len(output) * proportions[spidx]))
        # if test, include all remaining data (in case of rounding issues)
        if split == 'test':
            cursor_end = len(output)
        splits[split] = output[cursor:cursor_end]
        cursor = cursor_end
    print(f"Created random splits: {', '.join([f'{s}: {len(lines)}' for s, lines in splits.items()])} using random "
          f"seed {args.random_seed}.")

    # write splits to files
    for split, lines in splits.items():
        split_path = args.output_path + f'-{split}.csv'
        with open(split_path, 'w', encoding='utf8', newline='') as output_file:
            csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
            csv_writer.writerow(['text', 'label'])
            csv_writer.writerows(lines)
        print(f"Saved {split}-split with {len(lines)} items to '{split_path}'.")


if __name__ == '__main__':
    main()
