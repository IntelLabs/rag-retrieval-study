#!/bin/bash


python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot2_gold_default.yaml \
    --ndoc 5

python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot2_closedbook.py

k_vals=("0" "1" "2" "3" "4" "5" "10" "20" "100")

for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_bge_base_default.yaml \
        --ndoc ${k}
done

for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_colbert_default.yaml \
        --ndoc ${k}
done