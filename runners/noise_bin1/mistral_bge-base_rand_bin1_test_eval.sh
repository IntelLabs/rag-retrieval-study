#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

# ASQA
python3 reader/eval.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin0.json \
    --noise_first

python3 reader/eval.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin0.json


python3 reader/eval.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin1.json \
    --noise_first

python3 reader/eval.py \
    --f asqa--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin1.json

# QAMPARI

python3 reader/eval.py \
    --f qampari--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin0.json \
    --noise_first

python3 reader/eval.py \
    --f qampari--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin0.json


python3 reader/eval.py \
    --f qampari--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin1.json \
    --noise_first

python3 reader/eval.py \
    --f qampari--Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-bge-base-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin1.json

# NQ
python3 reader/eval_per_query.py \
    --f  nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-bge-base-bin0.json \
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file nq_bin1/nq_bin0.json \
    --nrand 5 \
    --citations

python3 reader/eval_per_query.py \
    --f  nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-bge-base-noise-first-bin0.json\
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file nq_bin1/nq_bin0.json \
    --nrand 5 \
    --noise_first \
    --citations

python3 reader/eval_per_query.py \
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-bge-base-bin1.json \
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file nq_bin1/nq_bin1.json \
    --nrand 5 \
    --citations

python3 reader/eval_per_query.py \
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-bge-base-noise-first-bin1.json \
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file nq_bin1/nq_bin1.json \
    --nrand 5 \
    --noise_first \
    --citations

