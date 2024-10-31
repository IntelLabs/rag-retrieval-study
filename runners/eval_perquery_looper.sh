#!/bin/bash
#SBATCH --job-name=pqeval
#SBATCH --partition=gpu-q
#SBATCH --nodes=1
#SBATCH -t 2-0
#SBATCH -o logs/eval_ndoc/%x-%A_%a.out


source setup/set_paths.sh

model_list=("llama" "mistral")
dataset_list=("asqa" "nq" "qampari")
# This will only run on the retrievers, not the closedbook and gold baselines
#ret_list=("bge-base" "colbert")
ret_list=("gold" "closedbook")

# Loop over k
#k_vals=("1" "2" "3" "4" "5" "10" "20" "100")
#for k in "${k_vals[@]}"; do
#    python3 reader/eval.py \
#        --f colbert/nq-Mistral-7B-Instruct-v0.3-None-shot0-ndoc${k}-42-colbert.json \
#        --no_bert
#done

big_array=()

for ret in "${ret_list[@]}"; do
    for model in "${model_list[@]}"; do
        for dataset in "${dataset_list[@]}"; do
            if [ "$dataset" = "asqa" ] || [ "$ret" = "gold" ] || [ "$ret" = "closedbook" ]; then
                folder="${dataset}_ndoc/${model}"
                file_list=$(ls results/reader/${folder}/*${ret}*.json | xargs -n 1 basename)
            elif [ "$dataset" = "nq" ] && [ "$ret" = "colbert" ]; then
                folder="${dataset}_ndoc/${model}/${ret}"
                file_list=$(ls results/reader/${folder}/*fixedgold.json | xargs -n 1 basename)
            else
                folder="${dataset}_ndoc/${model}/${ret}"
                file_list=$(ls results/reader/${folder}/*.json | xargs -n 1 basename)
            fi
            for fn in $file_list; do
                big_array+=("$folder/$fn")
                #echo "$folder/$fn"
                #args="--f $folder/$fn"
                #if [[ $fn == *"nq"* ]]; then
                #    args+=" --no_bert"
                #elif [[ $fn == *"cite"* ]]; then
                #    args+=" --citations"
                #fi
                #python3 reader/eval_per_query.py $args
            done
        done 
    done
done

echo ${big_array[@]}
echo ${#big_array[@]}

hostname
thisfile=${big_array[$SLURM_ARRAY_TASK_ID]}
python -u reader/eval_per_query.py --citations --f $thisfile
