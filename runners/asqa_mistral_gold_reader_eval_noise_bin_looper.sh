#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

percentile_vals=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")



for p in "${percentile_vals[@]}"; do
    python3 reader/eval.py \
        --f asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold-noise-${p}.json \
        --citations \
        --noise \
        --ndoc 5 \
        --nrand ${p} \
        --noise_file asqa_retrieval-bge-base-all-random-100.json
done

