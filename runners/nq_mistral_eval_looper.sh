#!/bin/bash
#SBATCH --job-name=nq_llama_ndoc_eval
#SBATCH --output=logs/llama_ndoc_eval-%j.out
#SBATCH --error=logs/llama_ndoc_eval-%j.err
#SBATCH --partition=gpu-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -t 4-0

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}
source ~/.bashrc
source setup/set_paths.sh
set -x

# finish nq_mistral generation
conda activate ragged
# no context 
python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_closedbook_default.yaml

# gold
python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_ndoc100_gold_default.yaml



conda deactivate
conda activate mauve-again

# no context 
python3 reader/eval_per_query.py \
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc0-42-cite-no-context.json \
    --no_bert \
    --citations

# gold
python3 reader/eval_per_query.py \ 
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold.json \
    --no_bert \
    --citations

python3 reader/eval_per_query.py \ 
    --f  nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc100-42-cite-gold.json \
    --no_bert \
    --citations

k_vals=("4" "5" "10" "20" "100")

for k in "${k_vals[@]}"; do
    python3 reader/eval_per_query.py \
        --f nq-Mistral-7B-Instruct-v0.3-cite-shot2-ndoc${k}-42-cite-bge-base.json \
        --no_bert \
        --citations
done

k_vals=("1" "2" "3" "4" "5" "10" "20" "100")


for k in "${k_vals[@]}"; do
    python3 reader/eval_per_query.py \
        --f nq-Mistral-7B-Instruct-v0.3-cite-shot2-ndoc${p}-42-cite-colbert.json \
        --no_bert \
        --citations
done



# no context 
python3 reader/eval.py \
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc0-42-cite-no-context.json \
    --no_bert \
    --citations

# gold
python3 reader/eval.py \ 
    --f nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc5-42-cite-gold.json \
    --no_bert \
    --citations

python3 reader/eval.py \ 
    --f  nq-Mistral-7B-Instruct-v0.3-None-shot2-ndoc100-42-cite-gold.json \
    --no_bert \
    --citations


for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f nq-Mistral-7B-Instruct-v0.3-cite-shot2-ndoc${k}-42-cite-bge-base.json \
        --no_bert \
        --citations
done


for k in "${k_vals[@]}"; do
    python3 reader/eval.py \
        --f nq-Mistral-7B-Instruct-v0.3-cite-shot2-ndoc${p}-42-cite-colbert.json \
        --no_bert \
        --citations
done
