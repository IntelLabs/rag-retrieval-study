#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}


# ASQA

python3 reader/eval.py \
    --f asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin0.json \
    --noise_first

python3 reader/eval.py \
    --f asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin0.json


python3 reader/eval.py \
    --f asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin1.json \
    --noise_first

python3 reader/eval.py \
    --f asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file asqa_bin1/asqa_bin1.json

# QAMPARI

python3 reader/eval.py \
    --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin0.json \
    --noise_first

python3 reader/eval.py \
    --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-bin0.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin0.json


python3 reader/eval.py \
    --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-noise-first-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin1.json \
    --noise_first

python3 reader/eval.py \
    --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-bge-base-bin1.json \
    --citations \
    --noise \
    --ndoc 5 \
    --nrand 5 \
    --noise_file qampari_bin1/qampari_bin1.json
