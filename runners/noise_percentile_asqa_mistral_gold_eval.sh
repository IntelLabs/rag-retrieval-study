#!/bin/bash

percentile_vals=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

for p in "${percentile_vals[@]}"; do
    python3 reader/eval.py \
        --f asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold-noise-${p}.json \
        --citations \
        --noise \
        --noise_first \
        --ndoc 5 \
        --nrand 5 \
        --noise_file asqa_retrieval-bge-base-all-random-${p}.json
done