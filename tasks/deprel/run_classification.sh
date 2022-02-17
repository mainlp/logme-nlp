#!/bin/bash

DATA_PATH=/home/max/data/ud29/deprel
EXP_PATH=/home/max/exp/logme/deprel
# Experiment Parameters
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
EMB_TYPE="transformer"
CLASSIFIER="mlp"
SEEDS=( 4012 5060 8823 8857 9908 )

# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    echo "Experiment: '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."

    exp_dir=$EXP_PATH/model${enc_idx}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
    # check if experiment already exists
    if [ -f "$exp_dir/best.pt" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
    # if experiment is new, train classifier
    else
      echo "Training ${CLASSIFIER}-classifier using '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."
      # train classifier
      python ../../classify.py \
        --task "token_classification" \
        --train_path $DATA_PATH/en-ewt-train.csv \
        --test_path $DATA_PATH/en-ewt-dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --classifier ${CLASSIFIER} \
        --seed ${SEEDS[$rsd_idx]}

      # save experiment info
      echo "${EMB_TYPE}:${ENCODERS[$enc_idx]} -> ${CLASSIFIER} with RS=${SEEDS[$rsd_idx]}" > $exp_dir/experiment-info.txt
    fi

    # check if prediction already exists
    if [ -f "$exp_dir/en-ewt-dev-pred.csv" ]; then
      echo "[Warning] Prediction '$exp_dir/en-ewt-dev-pred.csv' already exists. Not re-predicting."
    # if no prediction is available, run inference
    else
      # run prediction
      python ../../classify.py \
        --task "token_classification" \
        --train_path $DATA_PATH/en-ewt-train.csv \
        --test_path $DATA_PATH/en-ewt-dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --classifier ${CLASSIFIER} \
        --seed ${SEEDS[$rsd_idx]} \
        --prediction_only
    fi

    # run evaluation
    python ../../evaluate.py \
      --gold_path ${DATA_PATH}/en-ewt-dev.csv \
      --pred_path ${exp_dir}/en-ewt-dev-pred.csv \
      --out_path ${exp_dir}

    echo
  done
done
