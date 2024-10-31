#!/bin/bash
#SBATCH --job-name=bin1eval
#SBATCH --partition=gpu-p
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -o logs/eval_bin1/%x-%A_%a.out

# Set PYTHONPATH and other variables
source setup/set_paths.sh

# 4,8,11-15

datasets=("asqa" "nq")
retrievers=("gold" "bge-base")
model_strings=("Mistral-7B-Instruct-v0.3" "Llama-2-7b-chat-hf")
bin_id=("bin0" "bin1")
tag="None"

file_arr=()
noise_arr=()
for dataset in "${datasets[@]}"; do
    for model in "${model_strings[@]}"; do
        if [ "$model" = ${model_strings[0]} ]; then
            shot=2
        else
            shot=1
        fi
        for ret in "${retrievers[@]}"; do
            for bin in "${bin_id[@]}"; do
                filename=${dataset}_bin1/$dataset-$model-$tag-shot$shot-ndoc5-42-cite-$ret-$bin.json
                #filename=$RESULTS_PATH/reader/${dataset}_bin1/$dataset-$model-$tag-shot$shot-ndoc5-42-cite-$ret-$bin.json
                #ls $filename
                file_arr+=("$filename")
                noise_arr+=("noise_bins/${dataset}_${bin}.json")
            done
        done
    done
done


#echo ${#file_arr[@]}
#echo ${#noise_arr[@]}
thisnoise=${noise_arr[$SLURM_ARRAY_TASK_ID]}
thisfile=${file_arr[$SLURM_ARRAY_TASK_ID]}

python3 -u reader/eval_per_query.py --citations --noise --ndoc 5 --nrand 5 --noise_file $thisnoise --f $thisfile

