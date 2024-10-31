#!/bin/bash

percentile_vals=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

for p in "${percentile_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_gold_default.yaml \
        --eval_file asqa_eval_bge-large-test.json \
        --noise_file asqa_retrieval-bge-base-all-random-${p}.json \
        --nrand 5 \
        --noise_first \
        --add_name noise-${p}
done

