#!/bin/bash

source setup/set_path.sh

ret_list=("gold-recall")
#ret_list=("bge-base" "bge-large" "colbert")
#k_vals=("1" "2" "3" "4" "5" "10" "20" "100")
#for k in "${k_vals[@]}"; do
#    python3 reader/eval.py \
#        --f colbert/nq-Mistral-7B-Instruct-v0.3-None-shot0-ndoc${k}-42-colbert.json \
#        --no_bert
#done

for ret in "$ret_list"; do
    file_list=$(ls results/reader/${ret}/*.json | xargs -n 1 basename)
    for fn in $file_list; do
        echo "$ret/$fn"
        args="--f $ret/$fn"
        if [[ $fn == *"nq"* ]]; then
            args+=" --no_bert"
        elif [[ $fn == *"cite"* ]]; then
            args+=" --citations"
        fi
        python3 reader/eval_per_query.py $args 
    done
done
