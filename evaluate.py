#!/usr/bin/python3

import argparse
import csv
import json
import logging
import os

from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


def merge_files(gold_path: str, pred_path: str) -> str:
    name = pred_path.split(".")[0]
    out_path = f"{name}.merged.conll"

    with open(gold_path, "r") as gold_fp, open(pred_path, "r") as pred_fp, open(out_path, "w") as out_fp:

        for pred_line in pred_fp:
            gold_line = gold_fp.readline().strip().split("\t")

            if len(gold_line) < 2:
                out_fp.write("\n")
                continue

            pred_line = pred_line.strip().split("\t")
            out_fp.write(f"{gold_line[0]}\t{gold_line[1]}\t{pred_line[1]}\n")

    return name


def get_span_f1(gold_path: str, predicted_path: str) -> dict:
    # Merge gold and predictions
    merged_file = merge_files(gold_path, predicted_path)
    out = os.popen(f"perl project/src/utils/conlleval.perl -d '\t' < {merged_file}.merged.conll").read()
    # Delete merged file once read
    os.system(f"rm {merged_file}.merged.conll")

    if not out:
        exit(1)

    result = out.split()
    results_dict = {
            result[11]: float(result[12][:-2]),
            result[13]: float(result[14][:-2]),
            result[15]: float(result[16][:-2]),
            result[17]: float(result[18]),
            }

    return results_dict


def get_f1(gold_path: str, predicted_path: str) -> dict:
    gold = []
    pred = []

    with open(predicted_path) as pred_fp, open(gold_path) as gold_fp:
        gold_reader = csv.reader(gold_fp, delimiter=',')
        pred_reader = csv.reader(pred_fp, delimiter=',')
        for gold_line, pred_line in zip(gold_reader, pred_reader):
            gold.extend(gold_line[1].split(' '))
            pred.extend(pred_line[1].split(' '))

    assert len(gold) == len(pred), "Length of gold and predicted labels should be equal."

    return {"micro-F1"         : f1_score(gold, pred, average='micro') * 100,
            "macro-F1"         : f1_score(gold, pred, average='macro') * 100,
            "weighted-macro-F1": f1_score(gold, pred, average='weighted') * 100}


def main(args: argparse.Namespace):
    logging.info(f"Evaluating {args.gold_path} and {args.pred_path}.")
    exp = os.path.splitext(os.path.basename(args.pred_path))[0]

    if args.gold_path.endswith("conll") and args.pred_path.endswith("conll"):
        metrics = get_span_f1(args.gold_path, args.pred_path)
    elif args.gold_path.endswith("csv") and args.pred_path.endswith("csv"):
        metrics = get_f1(args.gold_path, args.pred_path)
    else:
        logging.info(f"File type is not supported, only (CSV or CONLL).")
        exit(1)

    logging.info(f"Saving scores to {args.out_path}")
    json.dump(metrics, open(f"{os.path.join(args.out_path, exp)}-results.json", "w"))

    logging.info(json.dumps(metrics, indent=4, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script for framework for LogME')

    parser.add_argument('--gold_path', type=str, nargs='?', required=True, help='Path to the gold labels FILE.')
    parser.add_argument('--pred_path', type=str, nargs='?', required=True, help='Path to the predicted labels FILE.')
    parser.add_argument('--out_path', type=str, nargs='?', required=True, help='Path where to save scores.')

    main(parser.parse_args())
