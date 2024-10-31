#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

k_vals=("1" "2" "3" "4" "5" "10" "20" "100")


for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config qampari_mistral-7b-instruct_shot2_colbert_default.yaml \
        --ndoc ${k}
done