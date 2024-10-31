#!/bin/bash
#SBATCH --job-name=nq_llama_nrand
#SBATCH --output=logs/nq_llama_nrand-%j.out
#SBATCH --error=logs/nq_llama_nrand-%j.err
#SBATCH -w bcl-gpu19
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -t 3-0

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}
source ~/.bashrc
source setup/set_paths.sh
set -x

conda activate ragged

# nrand_vals=("5" "10" "12")


# for p in "${nrand_vals[@]}"; do
#     python3 reader/run.py \
#         --config nq_llama-2-7b-chat_shot1_bge_base_default.yaml \
#         --ndoc 5 \
#         --noise_file nq_retrieval-bge-base-all-random-100.json \
#         --nrand ${p} \
#         --noise_first \
#         --add_name nrand-${p}-noise-100-shot1
# done

nrand_vals=("10" "12")

for p in "${nrand_vals[@]}"; do
    python3 reader/run.py \
        --config nq_llama-2-7b-chat_shot1_bge_base_default.yaml \
        --ndoc 5 \
        --noise_file nq_retrieval-bge-base-all-random-100.json \
        --nrand ${p} \
        --add_name nrand-${p}-noise-100-shot1
done