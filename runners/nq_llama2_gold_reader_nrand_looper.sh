#!/bin/bash
#SBATCH --job-name=asqa_llama_nrand
#SBATCH --output=logs/asqa_llama_nrand-%j.out
#SBATCH --error=logs/asqa_llama_nrand-%j.err
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

conda activate ragged
SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

# nrand_vals=("5" "10" "12")


# for p in "${nrand_vals[@]}"; do
#     python3 reader/run.py \
#         --config nq_llama-2-7b-chat_shot1_ndoc5_gold_default.yaml \
#         --ndoc 5 \
#         --noise_file nq_retrieval-bge-base-all-random-100.json \
#         --nrand ${p} \
#         --noise_first \
#         --add_name nrand-${p}-noise-100-shot1
# done

nrand_vals=("12")
for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config nq_llama-2-7b-chat_shot1_ndoc5_gold_default.yaml \
        --ndoc 5 \
        --noise_file nq_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --add_name nrand-${p}-noise-100-shot1
done