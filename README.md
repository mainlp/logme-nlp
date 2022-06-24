# LogME Framework

Anonymized code for **Evidence > Intuition: Transferability Estimation for Encoder Selection**.

This repository contains implementations to compute and evaluate the Logarithm of Maximum Evidence (LogME) on a wide variety of Natural Language Processing (NLP) tasks. It can be used to assess pre-trained models for transfer learning, where a pre-trained model with a high LogME value is likely to have good transfer performance (<a href="http://proceedings.mlr.press/v139/you21b/you21b.pdf">You et al., 2021</a>).

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
│   ├── tasks
│   │   ├── deidentification
│   │   │   ├── run_classification.sh
│   │   │   ├── run_classification_tuned.sh
│   │   │   └── run_logme.sh
│   │   ├── deprel
│   │   │   ├── convert.py
│   │   │   ├── run_classification.sh
│   │   │   └── run_logme.sh
│   │   ├── glue
│   │   │   ├── convert.py
│   │   │   ├── run_classification.sh
│   │   │   └── run_logme.sh
│   │   ├── sentiment
│   │   │   ├── convert.py
│   │   │   ├── run_classification.sh
│   │   │   └── run_logme.sh
│   │   ├── topic
│   │   │   ├── convert_news.py
│   │   │   ├── run_classification.sh
│   │   │   ├── run_classification_tuned.sh
│   │   │   └── run_logme.sh
│   │   ├── human
│   │   │   └── evaluate_rankings.py
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

## Usage
There are three main scripts used in all experiments:
```bash
# LogME Calculation for a dataset-LM pair
python main.py

# Classifier training using a dataset-LM pair
python classify.py

# Evaluation of predictions
python evaluate.py
```

For detailed usage, please refer to the examples below, and to the help output of each script:

```bash
python main.py -h
```

## Data

To run **LogME** on your data. The data needs to be pre-processed into a **.csv** format, where the labels must be converted to unique integers. If your dataset is available in <a href=https://huggingface.co/datasets>HuggingFace Datasets</a> you can use the name of the dataset in `main.py`.

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

Note that sequence labeling tasks require a pre-tokenized, space-separated input which has exactly as many tokens as labels.

## Experiments

Each experiment has a dedicated directory in `project/src/tasks/` containing a script for dataset conversion into the unified CSV-format (`convert.py`), LogME calculation (`run_logme.sh`), and classifier training and evaluation (`run_classification.sh`).

While many datasets are downloaded automatically, some require a separate, manual download (e.g., due to licensing). The tasks and corresponding datasets covered in the main paper are as follows:

* **AGNews (Zhang et al., 2015)** is a news topic classification dataset, the scripts for which can be found in `project/src/tasks/topic/`. The data is obtained from `huggingface`.
* **Airline Twitter (Crowdflower, 2020)** is a sentiment analysis dataset, the scripts for which can be found in `project/src/tasks/sentiment/`. It requires a separate download of the original data files.
* **SciERC (Luan et al., 2018)**
* **MNLI (Williams et al., 2018)** is a natural language inference dataset, the scripts for which can be found in `project/src/tasks/glue/`. The original data is downloaded automatically during the conversion process.
* **QNLI (Rajpurkar et al., 2016)** is a question answering / natural language inference dataset, the scripts for which can be found in `project/src/tasks/glue/`. The original data is downloaded automatically during the conversion process.
* **RTE (Giampiccolo et al., 2007)** is a natural language inference dataset, the scripts for which can be found in `project/src/tasks/glue/`. The original data is downloaded automatically during the conversion process.
* **EWT (Silveira et all., 2014)** is a syntactic dependency treebank, the scripts for which can be found in `project/src/tasks/sentiment/`. It requires a separate download of the original data files.
* **CrossNER (Liu et al., 2021)**
* **JobStack (Jensen et al., 2021)** is a deidentification of job postings dataset, the scripts for which can be found in `projects/src/tasks/deidentification/`. The data is obtained from the authors.

To run specific configurations of the experiments above, such as "mean-pooled sequence classification on BioBERT with full fine-tuning" etc., please refer to the examples below.

## Examples
For detailed example scripts check `project/tasks/*`.

### 1. Calculate LogME (example)
```bash
#!/bin/bash

# path to your data
DATA_PATH=project/resources/data/airline
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
python project/src/tasks/sentiment/convert.py $DATA_PATH/Tweets.csv $DATA_PATH/ -rs 4012

# iterate over encoders
for enc_idx in "${!ENCODERS[@]}"; do
  echo "Computing LogME using embeddings from '${ENCODERS[$enc_idx]}'"
  # compute embeddings and LogME
  python main.py \
    # sequence_classification OR sequence_labeling
    --task "sequence_classification" \
    --train_path $DATA_PATH/train.csv \
    --test_path $DATA_PATH/test.csv \
    # column headers in your .csv file
    --text_column text --label_column label \
    --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
    --pooling ${POOLING} | tee run_logme_cls.log
done
```

### 2. Model fine-tuning (example)
```bash
#!/bin/bash

DATA_PATH=project/resources/data/airline
EXP_PATH=project/output/sentiment
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
        --train_path $DATA_PATH/train.csv \
        --test_path $DATA_PATH/dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --pooling ${POOLING} \
        --classifier ${CLASSIFIER} \
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
      python classify.py \
        --task "sequence_classification" \
        --train_path $DATA_PATH/train.csv \
        --test_path $DATA_PATH/dev.csv \
        --exp_path ${exp_dir} \
        --embedding_model ${EMB_TYPE}:${ENCODERS[$enc_idx]} \
        --pooling ${POOLING} \
        --classifier ${CLASSIFIER} \
        --seed ${SEEDS[$rsd_idx]} \
        --prediction_only
    fi

    # run evaluation
    python evaluate.py \
      --gold_path ${DATA_PATH}/dev.csv \
      --pred_path ${exp_dir}/dev-pred.csv \
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
