#!/bin/bash


source setup/set_paths.sh
set -x

recallarray=(0.5 0.7 0.9)
model_id="mistral-7b-instruct"
nshot=2

for recall in "${recallarray[@]}"; do
    # NQ with citations
    python3 -u reader/run.py \
        --config nq_${model_id}_shot${nshot}_ndoc100_gold_default.yaml \
        --prompt_file nq_cite.json \
        --eval_file gold_recall/nq_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 100  # this just uses the maximum number of gold documents
    # NQ without citations
    python3 -u reader/run.py \
        --config nq_${model_id}_shot${nshot}_ndoc100_gold_default.yaml \
        --shot 0 \
        --prompt_file nq_default.json \
        --eval_file gold_recall/nq_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 100  # this just uses the maximum number of gold documents
    # ASQA with citations 
    python3 -u reader/run.py \
        --config asqa_${model_id}_shot${nshot}_gold_default.yaml \
        --eval_file gold_recall/asqa_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 5 \
done

