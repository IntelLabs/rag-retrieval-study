#!/bin/bash


source setup/set_paths.sh
set -x

recallarray=(0.7 0.9 0.95)
model_id="llama-2-7b-chat" #"mistral-7b-instruct"
model_short="llama"  #"mistral"
nshot=1 #2

for recall in "${recallarray[@]}"; do
    # NQ with citations
    python3 reader/run.py \
        --config nq_${model_id}_shot${nshot}_bge_base_default.yaml \
        --prompt_file ${dataset}_cite.json \
        --eval_file ${dataset}_retrieval_ann-recall${recall}_bge-base-dense.json \
        --add_name search-recall-${recall} \
        --ndoc 10
    # NQ without citations
    python3 reader/run.py \
        --config nq_${model_id}_shot${nshot}_bge_base_default.yaml \
        --shot 0 \
        --prompt_file ${dataset}_default.json \
        --eval_file ${dataset}_retrieval_ann-recall${recall}_bge-base-dense.json \
        --add_name search-recall-${recall} \
        --ndoc 10
    # ASQA with citations 
    python3 reader/run.py \
        --config ${dataset}_${model_id}_shot${nshot}_bge_base_default.yaml \
        --eval_file asqa_retrieval_ann-recall${recall}_bge-base-dense.json \
        --add_name search-recall-${recall} \
        --ndoc 10
done

