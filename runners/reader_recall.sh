#!/bin/bash
#SBATCH --job-name=nq_recall
#SBATCH --partition=gpu-a,gpu-p
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --exclude=bcl-gpu[11-12,15]

source setup/set_paths.sh

recall=(0.7 0.9)
dataset="nq"
retriever="base"
ndoc=10

#asqa_mistral-7b-instruct_shot2_ndoc10_bge_base_default.yaml
#asqa_retrieval_ann-recall0.9_bge-base-dense.json
#    --config ${dataset}_mistral-7b-instruct_shot2_ndoc10_bge_${retriever}_default.yaml \

#nq_mistral-7b-instruct_shot0_ndoc5_svs_default.yaml

r=${recall[$SLURM_ARRAY_TASK_ID]}
python3 -u reader/run.py \
    --config ${dataset}_mistral-7b-instruct_shot0_ndoc5_svs_default.yaml \
    --eval_file ${dataset}_retrieval_ann-recall${r}_bge-${retriever}-dense.json \
    --ndoc $ndoc &> logs/log_reader_${dataset}_bge-${retriever}_recall${r}_k${ndoc}.txt \
    --tag recall${r}
