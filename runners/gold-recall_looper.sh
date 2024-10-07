#!/bin/bash
#SBATCH --job-name=gold-recall
#SBATCH --partition=gpu-q
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH -t 2-0
#SBATCH -o logs/reader_gold-recall/%x-%A_%a.out

source setup/set_paths.sh
set -x

recallarray=(0.5 0.5 0.5 0.7 0.7 0.7 0.9 0.9 0.9)
recall=${recallarray[$SLURM_ARRAY_TASK_ID]}
model_id="llama-2-7b-chat"  #"mistral-7b-instruct"
model_short="llama"  #"mistral"
nshot=1  #2


if [ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 0 ]; then
    python3 -u reader/run.py \
        --config nq_${model_id}_shot${nshot}_ndoc100_gold_default.yaml \
        --prompt_file nq_cite.json \
        --eval_file gold_recall/nq_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 100 \
    &>> logs/log_reader_nq-cite_${model_short}_gold_recall${recall}.txt
elif [ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 1 ]; then
   python3 -u reader/run.py \
        --config asqa_${model_id}_shot${nshot}_gold_default.yaml \
        --eval_file gold_recall/asqa_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 5 \
    &>> logs/log_reader_asqa_${model_short}_gold_recall${recall}.txt
elif [ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 2 ]; then
    python3 -u reader/run.py \
        --config nq_${model_id}_shot${nshot}_ndoc100_gold_default.yaml \
        --shot 0 \
        --prompt_file nq_default.json \
        --eval_file gold_recall/nq_gold_recall${recall}.json \
        --add_name gold-recall-${recall} \
        --ndoc 100 \
    &>> logs/log_reader_nq-plain_${model_short}_gold_recall${recall}.txt
fi

