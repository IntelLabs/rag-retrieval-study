#!/bin/bash
#SBATCH --job-name=asqa_llama_per_query
#SBATCH --output=logs/asqa_llama_per_query-%j.out
#SBATCH --error=logs/asqa_llama_per_query-%j.err
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

# python3 reader/eval_per_query.py \
#     --f asqa_ndoc/llama/asqa-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-gold.json \
#     --citations

python3 reader/eval_per_query.py \
    --f asqa_ndoc/llama/asqa-Llama-2-7b-chat-hf-None-shot1-ndoc0-42-closedbook.json \
    --citations


k_vals=("1" "2" "3" "4" "5" "10" "20" "100")

for k in "${k_vals[@]}"; do

    python3 reader/eval_per_query.py \
        --f asqa_ndoc/llama/asqa-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-bge-base.json \
        --citations

done

for k in "${k_vals[@]}"; do

    python3 reader/eval_per_query.py \
        --f asqa_ndoc/llama/asqa-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-colbert.json \
        --citations

done