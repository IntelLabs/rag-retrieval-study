#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

k_vals=("1" "2" "3" "4" "5" "10" "20" "100")


for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
        --ndoc ${k}
done

for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_llama-2-7b-chat_shot1_colbert_default.yaml \
        --ndoc ${k}
done