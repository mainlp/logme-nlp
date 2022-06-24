#!/bin/bash

DATA_PATH=project/resources/data
EXP_PATH=project/resources/output/topic
# Experiment Parameters
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
EMB_TYPE="transformer+cls"
POOLING="first"
CLASSIFIER="mlp"
SEEDS=( 4012 5060 8823 8857 9908 )

# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    echo "Training ${CLASSIFIER}-classifier using '${ENCODERS[$enc_idx]}' (${POOLING}) and random seed ${SEEDS[$rsd_idx]}:"

    exp_dir=$EXP_PATH/model${enc_idx}-${POOLING}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
    # check if experiment already exists
    if [ -d "$exp_dir" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Skipped."
      continue
    fi

    # train classifier
    python3 classify.py \
      --train_path $DATA_PATH/agnews-train.csv \
      --test_path $DATA_PATH/agnews-dev.csv \
      --task 'sequence_classification' \
      --embedding_tuning \
      --batch_size 16 \
      --learning_rate 0.00005 \
      --exp_path ${exp_dir} \
      --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
      --pooling ${POOLING} \
      --classifier ${CLASSIFIER} \
      --seed ${SEEDS[$rsd_idx]}

    # save experiment info
    echo "${EMB_TYPE}:${ENCODERS[$enc_idx]} -> ${POOLING} -> ${CLASSIFIER} with RS=${SEEDS[$rsd_idx]}" > $exp_dir/experiment-info.txt
    echo
  done
done
