#!/bin/bash

DATA_PATH=project/resources/data/topic
EMB_TYPE="transformer"
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
POOLING="mean"

# prepare and split data
python3 project/tasks/topic/convert_news.py ag_news $DATA_PATH/agnews --text_column text --label_column label -rs 4012

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  echo "Computing LogME using embeddings from '${ENCODERS[$enc_idx]}'"
  # compute embeddings and LogME
  python3 main.py \
    --train_path $DATA_PATH/agnews-train.csv \
    --test_path $DATA_PATH/agnews-test.csv \
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
    --pooling ${POOLING} | tee run_logme_cls.log
done
