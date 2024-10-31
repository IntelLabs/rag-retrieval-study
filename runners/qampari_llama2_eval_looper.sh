#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

# gold 
python3 reader/eval.py \
    --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-gold.json \
    --citations

# no context
python3 reader/eval.py \
    --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc0-42-closedbook.json


k_vals=("1" "2" "3" "4" "5" "10" "20" "100")

for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-colbert.json \
        --citations
done

for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f qampari-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-bge-base.json \
        --citations
done
