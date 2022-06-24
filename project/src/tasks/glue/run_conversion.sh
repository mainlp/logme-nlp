#!/bin/bash

DATA_PATHS=( project/resources/data/glue/bert project/resources/data/glue/roberta )
TASKS="mnli qnli rte"
SEPS=( "[SEP]" "</s>" )

for sep_idx in "${!SEPS[@]}"; do
  python project/tasks/glue/convert.py $TASKS ${DATA_PATHS[$sep_idx]} -s "${SEP[$sep_idx]}"
done