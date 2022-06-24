#!/bin/bash

DATA_PATH=project/resources/data/scierc
#EMB_TYPE="transformer"
#POOLING="mean"
EMB_TYPE="transformer+cls"
POOLING="first"

ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  echo "Computing LogME using embeddings from '${ENCODERS[$enc_idx]}'"
  # compute embeddings and LogME
  python3 ../../main.py \
    --train_path $DATA_PATH/train.csv \
      --test_path $DATA_PATH/dev.csv \
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
    --pooling ${POOLING} | tee run_logme_cls.log
done