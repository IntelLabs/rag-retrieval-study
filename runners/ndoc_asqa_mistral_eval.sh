#!/bin/bash

k_vals=("0" "1" "2" "3" "4" "5" "10" "20" "100")

for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f  \
        --citation
done

python3 reader/eval_per_query.py \
    --f asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold.json\
    --citation

python3 reader/eval_per_query.py \
    --f asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc0-closedbook.json

k_vals=("0" "1" "2" "3" "4" "5" "10" "20" "100")

for k in "${k_vals[@]}"; do
    python3 reader/eval_per_query.py \
        --f  asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc${p}-42-cite-bge-base.json \
        --citation
done

for k in "${k_vals[@]}"; do
    python3 reader/eval_per_query.py \
        --f  asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc1-42-cite-colbert.json \
        --citation

done