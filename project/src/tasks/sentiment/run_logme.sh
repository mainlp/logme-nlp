#!/bin/bash

DATA_PATH=project/resources/data/airline
EMB_TYPE="transformer+cls"
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
POOLING="first"

# prepare and split data
python project/src/tasks/convert.py $DATA_PATH/Tweets.csv $DATA_PATH/ -rs 4012

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  echo "Computing LogME using embeddings from '${ENCODERS[$enc_idx]}'"
  # compute embeddings and LogME
  python main.py \
    --task "sequence_classification" \
    --train_path $DATA_PATH/train.csv \
    --test_path $DATA_PATH/test.csv \
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
    --pooling ${POOLING} | tee run_logme_cls.log
done
