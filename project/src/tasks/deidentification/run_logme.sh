#!/bin/bash

DATA_PATH=project/resources/data/jobstack
EMB_TYPE="transformer"
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  echo "Computing LogME using embeddings from '${ENCODERS[$enc_idx]}'"
  # compute embeddings and LogME
  python3 main.py \
    --task 'token_classification' \
    --train_path $DATA_PATH/train-jobstack.csv \
    --test_path $DATA_PATH/dev-jobstack.csv \
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} | tee run_logme.log
done