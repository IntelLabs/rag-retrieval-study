#!/bin/bash
#SBATCH --job-name=nq_llama_ndoc_eval
#SBATCH --output=logs/llama_ndoc_eval-%j.out
#SBATCH --error=logs/llama_ndoc_eval-%j.err
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

# # no context 
# python3 reader/eval_per_query.py \
#     --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc0-42-closedbook.json \
#     --no_bert \
#     --citations

# # gold
# python3 reader/eval_per_query.py \ 
#     --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-gold.json \
#     --no_bert \
#     --citations

# python3 reader/eval_per_query.py \ 
#     --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc100-42-cite-gold.json \
#     --no_bert \
#     --citations

k_vals=("1" "2" "3" "4" "5" "10")

for k in "${k_vals[@]}"; do
    python3 reader/eval_per_query.py \
        --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-bge-base.json \
        --no_bert \
        --citations
done

for k in "${k_vals[@]}"; do
    python3 reader/eval_per_query.py \
        --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-colbert.json \
        --no_bert \
        --citations
done



# no context 
python3 reader/eval.py \
    --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc0-42-closedbook.json \
    --no_bert \
    --citations

# gold
python3 reader/eval.py \ 
    --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc5-42-cite-gold.json \
    --no_bert \
    --citations

python3 reader/eval.py \ 
    --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc100-42-cite-gold.json \
    --no_bert \
    --citations

k_vals=("1" "2" "3" "4" "5" "10")

for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-bge-base.json \
        --no_bert \
        --citations
done

for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f nq-Llama-2-7b-chat-hf-None-shot1-ndoc${k}-42-cite-colbert.json \
        --no_bert \
        --citations
done
