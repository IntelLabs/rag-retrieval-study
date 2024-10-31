#!/bin/bash

python3 reader/eval_per_query.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin0.json \
    --noise_first

python3 reader/eval_per_query.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin0.json


python3 reader/eval_per_query.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin1.json \
    --noise_first

python3 reader/eval_per_query.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin1.json
