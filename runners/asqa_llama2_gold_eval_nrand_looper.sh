#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12")


for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-gold-noise-first-nrand-${p}-noise-100-shot1.json \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_first \
        --noise_file asqa_retrieval-bge-base-all-random-100.json \
        --citations
done


for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-gold-nrand-${p}-noise-100-shot1.json \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_file asqa_retrieval-bge-base-all-random-100.json \
        --citations
done