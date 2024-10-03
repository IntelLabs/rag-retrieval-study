#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12")


for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
        --ndoc 5 \
        --noise_file asqa_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --noise_first \
        --add_name nrand-${p}-noise-100-shot1
done


for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
        --ndoc 5 \
        --noise_file asqa_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --add_name nrand-${p}-noise-100-shot1
done