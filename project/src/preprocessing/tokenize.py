import argparse
import logging
from typing import List

from fasttext import tokenize
from transformers import AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


def tokenize_text(args: argparse.Namespace, X_train: List[str]) -> List[List[str]]:
    if args.tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, config=AutoConfig.from_pretrained(args.tokenizer))
            tokenized = [tokenizer.tokenize(sentence) for sentence in X_train]

        except OSError:
            logger.info("Cannot find model in HF, fallback to FT tokenization.")
            tokenized = [tokenize(sentence) for sentence in X_train]
    else:
        # assume it is tokenized
        tokenized = [sentence.split() for sentence in X_train]

    return tokenized
