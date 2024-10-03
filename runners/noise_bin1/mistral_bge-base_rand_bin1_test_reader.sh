#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}


python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin0.json \
    --nrand 5 \
    --add_name bin0

python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin0.json \
    --nrand 5 \
    --noise_first \
    --add_name bin0

python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin1.json \
    --nrand 5 \
    --add_name bin1

python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin1.json \
    --nrand 5 \
    --noise_first \
    --add_name bin1

