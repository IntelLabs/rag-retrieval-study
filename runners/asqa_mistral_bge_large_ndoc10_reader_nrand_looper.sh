#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

nrand_vals=("5" "10" "12" "15" "20")



for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_ndoc10_bge_base_default.yaml \
        --eval_file asqa_eval_bge-large-test.json \
        --noise_file asqa_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --noise_first \
        --add_name nrand-${p}-noise-100-test
done