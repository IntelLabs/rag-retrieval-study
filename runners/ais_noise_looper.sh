#!/bin/bash

#source setup/set_path.sh

ret="bge-base"
dataset="nq"
noise_order=("noise_first" "noise_last")

for order in "${noise_order[@]}"; do
    folder="results/workshop_data/${dataset}_nrand/$order"
    files=($(ls ${folder}/${dataset}*.json))
    for file in "${files[@]}"; do
        if [ -f $file ]; then
            python3 reader/eval_ais_perquery.py --f $file
        else
            echo "$file DOES NOT EXIST"
            exit 1
        fi
    done
done

