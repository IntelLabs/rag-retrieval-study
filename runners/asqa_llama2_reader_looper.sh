#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}


# no context 
python3 reader/run.py \
    --config asqa_llama-2-7b-chat_shot1_closedbook.yaml

# gold
python3 reader/run.py \
    --config asqa_llama-2-7b-chat_shot1_gold_default.yaml

k_vals=("1" "2" "3" "4" "5" "10")

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