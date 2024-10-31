#!/bin/bash


python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin0.json \
    --nrand 5 \
    --add_name bin0

python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin0.json \
    --nrand 5 \
    --noise_first \
    --add_name bin0

python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin1.json \
    --nrand 5 \
    --add_name bin1

python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot2_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin1.json \
    --nrand 5 \
    --noise_first \
    --add_name bin1

