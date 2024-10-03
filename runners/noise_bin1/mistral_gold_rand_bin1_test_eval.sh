#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

# ASQA
python3 reader/eval.py \
    --f asqa-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-bin0.json \
    --noise \
    --ndoc 5 \
    --noise_file asqa_bin1/asqa_bin0.json \
    --nrand 5 \
    --citations

python3 reader/eval.py \
    --f asqa-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-noise-first-bin0.json \
    --noise \
    --ndoc 5 \
    --noise_file asqa_bin1/asqa_bin0.json \
    --nrand 5 \
    --noise_first \
    --citations

python3 reader/eval.py \
    --f asqa-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-bin1.json \
    --noise \
    --ndoc 5 \
    --noise_file asqa_bin1/asqa_bin1.json \
    --nrand 5 \
    --citations

python3 reader/eval.py \
    --f asqa-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-noise-first-bin1.json \
    --noise \
    --ndoc 5 \
    --noise_file asqa_bin1/asqa_bin1.json \
    --nrand 5 \
    --noise_first \
    --citations

# QAMPARI
python3 reader/eval.py \
    --f qampari-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-bin0.json \
    --noise \
    --ndoc 5 \
    --noise_file qampari_bin1/qampari_bin0.json \
    --nrand 5 \
    --citations

python3 reader/eval.py \
    --f qampari-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-noise-first-bin0.json \
    --noise \
    --ndoc 5 \
    --noise_file qampari_bin1/qampari_bin0.json \
    --nrand 5 \
    --noise_first \
    --citations

python3 reader/eval.py \
    --f qampari-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-bin1.json \
    --noise \
    --ndoc 5 \
    --noise_file qampari_bin1/qampari_bin1.json \
    --nrand 5 \
    --citations


python3 reader/eval.py \
    --f qampari-Mistral-7B-Instruct-v0.3-None-shot1-ndoc5-42-cite-gold-noise-first-bin1.json \
    --noise \
    --ndoc 5 \
    --noise_file qampari_bin1/qampari_bin1.json \
    --nrand 5 \
    --noise_first \
    --citations

# NQ
python3 reader/eval_per_query.py \
    --f  nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold-bin0.json \
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin0.json \
    --nrand 5 \
    --citations

python3 reader/eval_per_query.py \
    --f  nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold-noise-first-bin0.json\
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin0.json \
    --nrand 5 \
    --noise_first \
    --citations

python3 reader/eval_per_query.py \
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold-bin1.json \
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin1.json \
    --nrand 5 \
    --citations


python3 reader/eval_per_query.py \
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold-noise-first-bin1.json \
    --no_bert \
    --noise \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin1.json \
    --nrand 5 \
    --noise_first \
    --citations

