#!/bin/bash

DATA_PATH=project/resources/data/scierc
EXP_PATH=project/resources/output/scierc
# Experiment Parameters
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )

#EMB_TYPE="transformer"
#POOLING="mean"
EMB_TYPE="transformer+cls"
POOLING="first"
CLASSIFIER="mlp"
SEEDS=( 4012 5060 8823 8857 9908 )
SPECIAL_TOKENS="[E1-START] [E2-START] [E1-END] [E2-END]"

# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    echo "Experiment: '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."

    exp_dir=$EXP_PATH/model${enc_idx}-tuned-${POOLING}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
    # check if experiment already exists
    if [ -f "$exp_dir/best.pt" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
    # if experiment is new, train classifier
    else
      echo "Training ${CLASSIFIER}-classifier using '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."
      # train classifier
      python3 ../../classify.py \
        --task "sequence_classification" \
        --train_path $DATA_PATH/train.csv \
        --test_path $DATA_PATH/dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --pooling ${POOLING} \
        --classifier ${CLASSIFIER} \
        --embedding_tuning \
        --learning_rate 0.00005 \
        --special_tokens $SPECIAL_TOKENS \
        --seed ${SEEDS[$rsd_idx]}

      # save experiment info
      echo "${EMB_TYPE}:${ENCODERS[$enc_idx]} -> ${POOLING} -> ${CLASSIFIER} with RS=${SEEDS[$rsd_idx]}" > $exp_dir/experiment-info.txt
    fi

    # check if prediction already exists
    if [ -f "$exp_dir/dev-pred.csv" ]; then
      echo "[Warning] Prediction '$exp_dir/dev-pred.csv' already exists. Not re-predicting."
    # if no prediction is available, run inference
    else
      # run prediction
      python3 ../../classify.py \
        --task "sequence_classification" \
        --train_path $DATA_PATH/train.csv \
        --test_path $DATA_PATH/dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --embedding_tuning \
        --pooling ${POOLING} \
        --classifier ${CLASSIFIER} \
        --seed ${SEEDS[$rsd_idx]} \
        --special_tokens $SPECIAL_TOKENS \
        --prediction_only
    fi

    # run evaluation
    python3 ../../evaluate.py \
      --gold_path ${DATA_PATH}/dev.csv \
      --pred_path ${exp_dir}/dev-pred.csv \
      --out_path ${exp_dir}

    echo
  done
done
