#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12" "15" "20")

for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config qampari_llama-2-7b-chat-shot1_bge_base_default.yaml \
        --ndoc 5 \
        --noise_file qampari_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --add_name nrand-${p}-noise-100-final \
        --noise_first
done

for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config qampari_llama-2-7b-chat-shot1_bge_base_default.yaml \
        --ndoc 5 \
        --noise_file qampari_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --add_name nrand-${p}-noise-100-final
done