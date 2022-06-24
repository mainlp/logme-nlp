#!/bin/bash

DATA_PATH=project/resources/data/glue
TASKS=( "mnli" "qnli" "rte" )
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
EMB_TYPE="transformer+cls"
POOLING="first"

# iterate over tasks
for tsk_idx in "${!TASKS[@]}"; do
  task=${TASKS[$tsk_idx]}
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    encoder=${ENCODERS[$enc_idx]}
    data_dir=$DATA_PATH
    echo "Computing LogME using embeddings from '$EMB_TYPE:$encoder' for task '$task'."

    # point to data dir with appropriate SEP token
    if [[ $encoder == "roberta-base" ]] || [[ $encoder == "cardiffnlp/twitter-roberta-base" ]]; then
      data_dir=$data_dir/roberta
    else
      data_dir=$data_dir/bert
    fi

    # set up training and validation paths
    train_path=$data_dir/$task-train.csv
    valid_paths=( $data_dir/$task-validation.csv )
    # special case: MNLI
    if [[ $task == "mnli" ]]; then
#      valid_paths=( $data_dir/$task-validation_matched.csv valid_path=$data_dir/$task-validation_mismatched.csv )
      valid_paths=( $data_dir/$task-validation_matched.csv )
    fi

    # iterate over validation paths
    for vld_idx in "${!valid_paths[@]}"; do
      valid_path=${valid_paths[$vld_idx]}
      # compute embeddings and LogME
      python main.py \
        --task "sequence_classification" \
        --train_path $train_path \
        --test_path $valid_path \
        --text_column text --label_column label \
        --embedding_model ${EMB_TYPE}:${encoder} \
        --pooling ${POOLING} | tee run_logme_cls.log
    done
  done
done
