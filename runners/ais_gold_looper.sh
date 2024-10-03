#!/bin/bash

#source setup/set_path.sh

ret="bge-base"

datasets=("asqa" "nq")
#datasets=("nq")
models=("llama" "mistral")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        folder="results/workshop_data/${dataset}_ndoc/${model}/"
        files=($(ls ${folder}*cite-gold*.json))
        for targetfile in "${files[@]}"; do
            if [[ -f "$targetfile" ]]; then
                python3 reader/eval_ais_perquery.py --f $targetfile
            else
                echo "${targetfile} DOES NOT EXIST"
                exit 1
            fi
        done
    done
done

