#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

python3 reader/run.py \
    --config qampari_mistral-7b-instruct_shot2_gold_default.yaml

python3 reader/run.py \
    --config qampari_mistral-7b-instruct_shot2_closedbook_default.yaml

k_vals=("1" "2" "3" "4" "5" "10" "20" "100")


for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config qampari_mistral-7b-instruct_shot2_bge_base_default.yaml \
        --ndoc ${k}
done

for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config qampari_mistral-7b-instruct_shot2_colbert_default.yaml \
        --ndoc ${k}
done