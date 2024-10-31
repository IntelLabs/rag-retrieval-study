#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12" "15" "20")




for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-bge-base-noise-first-nrand-${p}-noise-100-final.json\
        --citations \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_file asqa_retrieval-bge-base-all-random-100.json \
        --noise_first
done

for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-bge-base-nrand-${p}-noise-100-final.json \
        --citations \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_file asqa_retrieval-bge-base-all-random-100.json
done