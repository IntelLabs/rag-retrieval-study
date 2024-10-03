#!/bin/bash
#SBATCH --job-name=gold-recall
#SBATCH --partition=gpu-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200GB
#SBATCH -t 2-0

source setup/set_paths.sh
set -x

recallarray=(0.5 0.5 0.5 0.7 0.7 0.7 0.9 0.9 0.9)
recall=${recallarray[$SLURM_ARRAY_TASK_ID]}


if [ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 0 ]; then
    python3 -u reader/run.py \
        --config nq_mistral-7b-instruct_shot2_ndoc100_gold_default.yaml \
        --prompt_file nq_cite.json \
        --eval_file gold_recall/nq_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 100 \
    &>> logs/log_reader_nq-cite_mistral_gold_recall${recall}.txt
elif [ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 1 ]; then
   python3 -u reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_gold_default.yaml \
        --eval_file gold_recall/asqa_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 5 \
    &>> logs/log_reader_asqa_mistral_gold_recall${recall}.txt
elif [ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 2 ]; then
    python3 -u reader/run.py \
        --config nq_mistral-7b-instruct_shot2_ndoc100_gold_default.yaml \
        --shot 0 \
        --prompt_file nq_default.json \
        --eval_file gold_recall/nq_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 100 \
    &>> logs/log_reader_nq-plain_mistral_gold_recall${recall}.txt
fi

