#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12" "15" "20")


for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-noise-first-nrand-${p}-noise-100-shot1.json \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_first \
        --noise_file qampari_retrieval-bge-base-all-random-100.json \
        --citations
        
done


for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-nrand-${p}-noise-100-shot1.json \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_file qampari_retrieval-bge-base-all-random-100.json \
        --citations
        
done