# LogME Framework

Code for Evidence > Intuition: Transferability Estimation for Encoder Selection.

The Logarithm of Maximum Evidence (LogME) can be used to assess pre-trained models for transfer learning: a pre-trained 
model with a high LogME value is likely to have good transfer performance
(<a href="http://proceedings.mlr.press/v139/you21b/you21b.pdf">You et al., 2021</a>).

## Project Structure
```
project
├── resources (run setup.sh and add data)
│   ├── data (run setup.sh and add data)
│   │   └── *
│   ├── output (run setup.sh and add data)
│   │   └── * 
├── src
│   ├── classification
│   │   ├── __init__.py
│   │   ├── classifiers.py
│   │   └── losses.py
│   ├── preprocessing
│   │   └── tokenize.py
│   ├── utils
│   │   ├── conll_2_string.py
│   │   ├── string_2_conll.py
│   │   ├── conlleval.perl
│   │   ├── data.py
│   │   ├── embeddings.py
│   │   ├── encode_data.py
│   │   ├── leep.py (deprecated)
│   │   ├── load_data.py
│   │   └── logme.py
├── tasks
│   ├── deprel
│   │   ├── convert_ud.py
│   │   ├── run_classification.sh
│   │   └── run_logme.sh
│   ├── sentiment
│   │   ├── convert_airline.py
│   │   ├── run_classification.sh
│   │   └── run_logme.sh
│   ├── human
│   │   └── evaluate_rankings.py
├── .gitignore
├── classify.py
├── evaluate.py
├── main.py
├── README.md
├── requirements.txt
└── setup.sh
```

## Requirements
```
numpy
scipy
sklearn
torch
transformers
datasets
numba
```
```bash
pip install --user -r requirements.txt
```

#### Setup
Run `bash setup.sh` to create the appropriate directory paths.

## Usages
There are three scripts to use:
```python
main.py
classify.py
evaluate.py

# check usage by e.g.
python main.py -h
```

## Data

To run **LogME** on your data. The data needs to be pre-tokenized in a **.csv** format, where the labels must be 
converted to 
unique 
integers:

#### Sequence Classification
```csv
"text","label"
"this is a sentence , to test .","0"
...
```

#### Sequence Labeling
```csv
"text","label"
"this is New York .","0 0 1 2 0"
...
```

If your dataset is available in <a href=https://huggingface.co/datasets>HuggingFace Datasets</a> you can use the 
name of the dataset in `main.py`.

## Examples
For detailed example scripts check `project/tasks/*`.

### 1. Calculate LogME (example)
```bash
#!/bin/bash

# path to your data
DATA_PATH=~/project/resources/data/airline
# the type of embedding to calculate LogME on (e.g., [cls]-token or the mean of subwords) 
# [transformer, transformer+cls]
EMB_TYPE="transformer+cls"
# your favourite encoders to vectorize your data with.
ENCODERS=( "bert-base-uncased" 
           "roberta-base"
           "distilbert-base-uncased" 
           "emilyalsentzer/Bio_ClinicalBERT" 
           "dmis-lab/biobert-v1.1" 
           "cardiffnlp/twitter-roberta-base" 
           "allenai/scibert_scivocab_uncased" )
# use POOLING="first" if you calculate LogME over the [cls] token, otherwise "mean" is default.
POOLING="first"

# prepare and split data
python convert_airline.py $DATA_PATH/Tweets.csv $DATA_PATH/notok -rs 4012

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  echo "Computing LogME using embeddings from '${ENCODERS[$enc_idx]}'"
  # compute embeddings and LogME
  python main.py \
    # sequence_classification OR sequence_labeling
    --task "sequence_classification" \
    --train_path $DATA_PATH/notok-train.csv \
    --test_path $DATA_PATH/notok-test.csv \
    # column headers in your .csv file
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
    --pooling ${POOLING} | tee run_logme_cls.log
done
```

### 2. Model fine-tuning (example)
```
#!/bin/bash

DATA_PATH=~/project/resources/data/airline
EXP_PATH=~/project/exp/logme/airline
# Experiment Parameters
ENCODERS=( "bert-base-uncased" "roberta-base" "distilbert-base-uncased" "emilyalsentzer/Bio_ClinicalBERT" "dmis-lab/biobert-v1.1" "cardiffnlp/twitter-roberta-base" "allenai/scibert_scivocab_uncased" )
#EMB_TYPE="transformer"
#POOLING="mean"
EMB_TYPE="transformer+cls"
POOLING="first"
CLASSIFIER="mlp"
SEEDS=( 4012 5060 8823 8857 9908 )

# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!ENCODERS[@]}"; do
    echo "Experiment: '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."

    exp_dir=$EXP_PATH/model${enc_idx}-${POOLING}-${CLASSIFIER}-rs${SEEDS[$rsd_idx]}
    # check if experiment already exists
    if [ -f "$exp_dir/best.pt" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
    # if experiment is new, train classifier
    else
      echo "Training ${CLASSIFIER}-classifier using '${ENCODERS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."
      # train classifier
      python classify.py \
        --task "sequence_classification" \
        --train_path $DATA_PATH/notok-train.csv \
        --test_path $DATA_PATH/notok-dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --pooling ${POOLING} \
        --classifier ${CLASSIFIER} \
        --seed ${SEEDS[$rsd_idx]}

      # save experiment info
      echo "${EMB_TYPE}:${ENCODERS[$enc_idx]} -> ${POOLING} -> ${CLASSIFIER} with RS=${SEEDS[$rsd_idx]}" > $exp_dir/experiment-info.txt
    fi

    # check if prediction already exists
    if [ -f "$exp_dir/notok-dev-pred.csv" ]; then
      echo "[Warning] Prediction '$exp_dir/notok-dev-pred.csv' already exists. Not re-predicting."
    # if no prediction is available, run inference
    else
      # run prediction
      python classify.py \
        --task "sequence_classification" \
        --train_path $DATA_PATH/notok-train.csv \
        --test_path $DATA_PATH/notok-dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --pooling ${POOLING} \
        --classifier ${CLASSIFIER} \
        --seed ${SEEDS[$rsd_idx]} \
        --prediction_only
    fi

    # run evaluation
    python evaluate.py \
      --gold_path ${DATA_PATH}/notok-dev.csv \
      --pred_path ${exp_dir}/notok-dev-pred.csv \
      --out_path ${exp_dir}

    echo
  done
done
```

### 3. Evaluation (example)
```bash
# path to your data
DATA_PATH=~/project/resources/data/jobstack
EXP_DIR=~/project/resources/output/jobstack

# convert predictions to conll if you do sequence labeling and you have data in conll format
python project/src/utils/string_2_conll.py \
  --input ${EXP_DIR}/jobstack-predictions.csv \
  --output ${EXP_DIR}/jobstack-predictions.conll \
  --labels ${DATA_PATH}/labels.json \

# run evaluation, in this example on dev.
python evaluate.py \
  --gold_path ${DATA_PATH}/dev-jobstack.conll \
  --pred_path ${EXP_DIR}/jobstack-predictions.conll \
  --out_path ${EXP_DIR}
```
