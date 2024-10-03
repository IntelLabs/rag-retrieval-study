#!/bin/bash
#SBATCH --job-name=asqa_mistral_per_query
#SBATCH --output=logs/asqa_mistral_per_query-%j.out
#SBATCH --error=logs/asqa_mistral_per_query-%j.err
#SBATCH --partition=gpu-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -t 2-0

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}
source ~/.bashrc
source setup/set_paths.sh
set -x

conda activate mauve-again

python3 reader/eval_per_query.py \
    --f asqa_ndoc/mistral/asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc0-closedbook.json\
    --citations

python3 reader/eval_per_query.py \
    --f asqa_ndoc/mistral/asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold.json \
    --citations


k_vals=("1" "2" "3" "4" "5" "10")

for k in "${k_vals[@]}"; do

    python3 reader/eval_per_query.py \
        --f asqa_ndoc/mistral/asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc${k}-42-cite-bge-base.json \
        --citations

done

for k in "${k_vals[@]}"; do

    python3 reader/eval_per_query.py \
        --f asqa_ndoc/mistral/asqa-Mistral-7B-Instruct-v0.3-None-shot2-ndoc${k}-42-cite-colbert.json \
        --citations

done