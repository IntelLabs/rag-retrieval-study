#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

percentile_vals=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

for p in "${percentile_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_ndoc10_bge_base_gold_noise.yaml \
        --noise_file asqa_retrieval-bge-base-all-test-random-${p}.json \
        --add_name noise-${p}
done