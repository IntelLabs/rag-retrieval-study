#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12")


for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f  \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_first \
        --noise_file nq_retrieval-bge-base-all-random-100.json \
        --citations
        
done


for p in "${nrand_vals[@]}"; do
    python3 reader/eval.py \
        --f  \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_file nq_retrieval-bge-base-all-random-100.json \
        --citations
done