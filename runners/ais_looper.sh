#!/bin/bash

#source setup/set_path.sh

#ret="colbert"
ret="bge-base"

#datasets=("asqa" "nq")
datasets=("nq")
models=("llama" "mistral")
#models=("mistral")
k_vals=("1" "5" "10")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        if [ "$model" = "llama" ]; then
            model_str='Llama-2-7b-chat-hf'
            tag="None-shot1"
        elif [ "$model" = "mistral" ]; then
            model_str='Mistral-7B-Instruct-v0.3'
            if [ "$dataset" = "nq" ]; then
                tag="cite-shot2"
            else
                tag="None-shot2"
            fi
        fi 
        folder="results/workshop_data/${dataset}_ndoc/${model}/${ret}/"
        cpdir="/home/aleto/projects/rag-svs/alce/results/reader/${dataset}_ndoc/${model}"
        for k in "${k_vals[@]}"; do
            base_file="$dataset-$model_str-$tag-ndoc$k-42-cite-${ret}.json"
            target_file="${folder}${base_file}"
            if [ -f "$target_file" ]; then
                python3 reader/eval_ais_perquery.py --f $target_file
            else
                echo "$target_file DOES NOT EXIST, copying..."
                cp $cpdir/$base_file $folder
                python3 reader/eval_ais_perquery.py --f $target_file
                #exit 1
            fi
        done
    done
done

