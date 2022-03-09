#!/usr/bin/python3

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from project.src.classification import load_classifier
from project.src.utils.data import LabelledDataset
from project.src.utils.embeddings import load_embeddings, load_pooling_function
# local imports
from project.src.utils.load_data import get_dataset


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Classifier Training')

    # data setup
    arg_parser.add_argument('--train_path', help='path to training data')
    arg_parser.add_argument('--test_path', help='path to validation data')
    arg_parser.add_argument('--dataset', help='name of HuggingFace dataset')
    arg_parser.add_argument('--task', choices=['sequence_classification', 'token_classification'],
                            help='''Specify the type of task. Token classification requires pre-tokenized text and 
                            one label per token (both separated by space). Sequence classification requires pooling 
                            to reduce a sentence's token embeddings to one embedding per sentence.''')
    arg_parser.add_argument('-st', '--special_tokens', nargs='*', help='special tokens list')
    arg_parser.add_argument('--text_column', default='text', help='column containing input features')
    arg_parser.add_argument('--label_column', default='label', help='column containing gold labels')

    # embedding model setup
    arg_parser.add_argument('--embedding_model', required=True, help='embedding model identifier')
    arg_parser.add_argument('-pl', '--pooling', help='pooling strategy for sentence classification (default: None)')
    arg_parser.add_argument('-et', '--embedding_tuning', action='store_true', default=False,
                            help='set flag to tune the full model including embeddings (default: False)')

    # classifier setup
    arg_parser.add_argument('--classifier', required=True, help='classifier identifier')
    arg_parser.add_argument('-po', '--prediction_only', action='store_true', default=False,
                            help='set flag to run prediction on the validation data and exit (default: False)')

    # experiment setup
    arg_parser.add_argument('--exp_path', required=True, help='path to experiment directory')
    arg_parser.add_argument('-e', '--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
    arg_parser.add_argument('-es', '--early_stop', type=int, default=3,
                            help='maximum number of epochs without improvement (default: 3)')
    arg_parser.add_argument('-bs', '--batch_size', type=int, default=32,
                            help='maximum number of sentences per batch (default: 32)')
    arg_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    arg_parser.add_argument('-rs', '--seed', type=int, help='seed for probabilistic components (default: None)')

    return arg_parser.parse_args()


def setup_experiment(out_path, prediction=False):
    if not os.path.exists(out_path):
        if prediction:
            print(f"Experiment path '{out_path}' does not exist. Cannot run prediction. Exiting.")
            exit(1)

        # if output dir does not exist, create it (new experiment)
        print(f"Path '{out_path}' does not exist. Creating...")
        os.mkdir(out_path)
    # if output dir exist, check if predicting
    else:
        # if not predicting, verify overwrite
        if not prediction:
            response = None

            while response not in ['y', 'n']:
                response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
            if response == 'n':
                exit(1)

    # setup logging
    log_format = '%(message)s'
    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(out_path, 'classify.log'), filemode='a', format=log_format,
                        level=log_level)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def run(classifier, criterion, optimizer, dataset, batch_size, mode='train', return_predictions=False):
    stats = defaultdict(list)

    # set model to training mode
    if mode == 'train':
        classifier.train()
        batch_generator = dataset.get_shuffled_batches
    # set model to eval mode
    elif mode == 'eval':
        classifier.eval()
        batch_generator = dataset.get_batches

    # iterate over batches
    for bidx, batch_data in enumerate(batch_generator(batch_size)):
        # set up batch data
        sentences, labels, num_remaining = batch_data

        # when training, perform both forward and backward pass
        if mode == 'train':
            # zero out previous gradients
            optimizer.zero_grad()

            # forward pass
            predictions = classifier(sentences)

            # propagate loss
            loss = criterion(predictions['flat_logits'], labels)
            loss.backward()
            optimizer.step()

        # when evaluating, perform forward pass without gradients
        elif mode == 'eval':
            with torch.no_grad():
                # forward pass
                predictions = classifier(sentences)
                # calculate loss
                loss = criterion(predictions['flat_logits'], labels)

        # calculate accuracy
        accuracy = criterion.get_accuracy(predictions['flat_logits'].detach(), labels)

        # store statistics
        stats['loss'].append(float(loss.detach()))
        stats['accuracy'].append(float(accuracy))

        # store predictions
        if return_predictions:
            # iterate over inputs items
            for sidx in range(predictions['labels'].shape[0]):
                # append non-padding predictions as list
                predicted_labels = predictions['labels'][sidx]
                stats['predictions'].append(predicted_labels[predicted_labels != -1].tolist())

        # print batch statistics
        pct_complete = (1 - (num_remaining / len(dataset._inputs))) * 100
        sys.stdout.write(
                f"\r[{mode.capitalize()} | Batch {bidx + 1} | {pct_complete:.2f}%] "
                f"Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f}"
                )
        sys.stdout.flush()

    # clear line
    print("\r", end='')

    return stats


def main():
    args = parse_arguments()

    # setup experiment directory and logging
    setup_experiment(args.exp_path, prediction=args.prediction_only)

    if args.prediction_only: logging.info(f"Running in prediction mode (no training).")

    # set random seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # TODO HuggingFace Datasets integration
    train_sentences, train_labels, valid_sentences, valid_labels = get_dataset(args)

    # setup data
    train_data = LabelledDataset(inputs=train_sentences, labels=train_labels)
    logging.info(f"Loaded {train_data} (train).")
    valid_data = LabelledDataset(inputs=valid_sentences, labels=valid_labels)
    logging.info(f"Loaded {valid_data} (dev).")
    # gather labels
    if set(train_data.get_label_types()) < set(valid_data.get_label_types()):
        logging.warning(f"[Warning] Validation data contains labels unseen in the training data.")
    label_types = sorted(set(train_data.get_label_types()) | set(valid_data.get_label_types()))

    # load embedding model
    embedding_model = load_embeddings(
            args.embedding_model,
            tokenized=(args.task == 'token_classification'),
            static=(not args.embedding_tuning),
            special_tokens=args.special_tokens
            )
    logging.info(f"Loaded {embedding_model}.")

    # load pooling function for sentence labeling tasks
    pooling_function = None
    if args.pooling is not None:
        pooling_function = load_pooling_function(args.pooling)
        logging.info(f"Applying pooling function '{args.pooling}' to token embeddings.")

    # load classifier and loss constructors based on identifier
    classifier_constructor, loss_constructor = load_classifier(args.classifier)

    # setup classifier
    classifier = classifier_constructor(
            emb_model=embedding_model, emb_pooling=pooling_function, emb_tuning=args.embedding_tuning,
            classes=label_types
            )
    logging.info(f"Using classifier:\n{classifier}")
    # load pre-trained model for prediction
    if args.prediction_only:
        classifier_path = os.path.join(args.exp_path, 'best.pt')
        if not os.path.exists(classifier_path):
            logging.error(f"[Error] No pre-trained model available in '{classifier_path}'. Exiting.")
            exit(1)
        classifier = classifier_constructor.load(
            classifier_path, classes=label_types,
            emb_model=embedding_model, emb_pooling=pooling_function, emb_tuning=args.embedding_tuning
        )
        logging.info(f"Loaded pre-trained classifier from '{classifier_path}'.")

    # setup loss
    criterion = loss_constructor(label_types)
    logging.info(f"Using criterion {criterion}.")

    # main prediction call (when only predicting on validation data w/o training)
    if args.prediction_only:
        stats = run(
            classifier, criterion, None, valid_data,
            args.batch_size, mode='eval', return_predictions=True
        )
        # convert label indices back to string labels
        idx_lbl_map = {idx: lbl for idx, lbl in enumerate(label_types)}
        pred_labels = [
            [idx_lbl_map[p] for p in preds]
            for preds in stats['predictions']
        ]
        pred_data = LabelledDataset(valid_data._inputs, pred_labels)
        pred_path = os.path.join(args.exp_path, f'{os.path.splitext(os.path.basename(args.test_path))[0]}-pred.csv')
        pred_data.save(pred_path)
        logging.info(f"Prediction completed with Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f} (mean over batches).")
        logging.info(f"Saved results from {pred_data} to '{pred_path}'. Exiting.")
        exit()

    # setup optimizer
    optimizer = torch.optim.AdamW(params=classifier.get_trainable_parameters(), lr=args.learning_rate)
    logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")

    # main loop
    stats = defaultdict(list)
    for ep_idx in range(args.epochs):
        # iterate over training batches and update classifier weights
        ep_stats = run(
                classifier, criterion, optimizer, train_data, args.batch_size, mode='train'
                )
        # print statistics
        logging.info(
                f"[Epoch {ep_idx + 1}/{args.epochs}] Train completed with "
                f"Acc: {np.mean(ep_stats['accuracy']):.4f}, Loss: {np.mean(ep_stats['loss']):.4f}"
                )

        # iterate over batches in dev split
        ep_stats = run(
                classifier, criterion, None, valid_data, args.batch_size, mode='eval'
                )

        # store and print statistics
        for stat in ep_stats:
            stats[stat].append(np.mean(ep_stats[stat]))
        logging.info(
                f"[Epoch {ep_idx + 1}/{args.epochs}] Validation completed with "
                f"Acc: {stats['accuracy'][-1]:.4f}, Loss: {stats['loss'][-1]:.4f}"
                )
        cur_eval_loss = stats['loss'][-1]

        # save most recent model
        path = os.path.join(args.exp_path, 'newest.pt')
        classifier.save(path)
        logging.info(f"Saved model from epoch {ep_idx + 1} to '{path}'.")

        # save best model
        if cur_eval_loss <= min(stats['loss']):
            path = os.path.join(args.exp_path, 'best.pt')
            classifier.save(path)
            logging.info(f"Saved model with best loss {cur_eval_loss:.4f} to '{path}'.")

        # check for early stopping
        if (ep_idx - stats['loss'].index(min(stats['loss']))) >= args.early_stop:
            logging.info(f"No improvement since {args.early_stop} epochs ({min(stats['loss']):.4f} loss). Early stop.")
            break

    logging.info(f"Training completed after {ep_idx + 1} epochs.")


if __name__ == '__main__':
    main()
