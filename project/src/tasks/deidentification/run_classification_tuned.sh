#!/bin/bash

DATA_PATH=project/resources/data/jobstack
EXP_PATH=project/resources/output/jobstack
# Experiment Parameters
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
EMB_TYPE="transformer"
CLASSIFIER="mlp"
SEEDS=( 4012 5060 8823 8857 9908 )

# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    echo "Training ${CLASSIFIER}-classifier using '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}:"

    exp_dir=$EXP_PATH/model${enc_idx}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
    # check if experiment already exists
    if [ -d "$exp_dir" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Skipped."
      continue
    fi

    # train classifier
    python classify.py \
      --task "token_classification" \
      --train_path $DATA_PATH/train-jobstack.csv \
      --test_path $DATA_PATH/dev-jobstack.csv \
      --embedding_tuning \
      --learning_rate 0.00005 \
      --batch_size 16 \
      --exp_path ${exp_dir} \
      --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
      --classifier ${CLASSIFIER} \
      --seed ${SEEDS[$rsd_idx]}

    # save experiment info
    echo "${EMB_TYPE}:${ENCODERS[$enc_idx]} -> ${CLASSIFIER} with RS=${SEEDS[$rsd_idx]}" > $exp_dir/experiment-info.txt
    echo
  done
done