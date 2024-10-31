#!/bin/bash
#SBATCH --job-name=readnq-colbert
#SBATCH --nodes=1
#SBATCH -t 3-0
#SBATCH -o logs/reader_ndoc/%x-%A_%a.out


###
# The following command would launch 5 jobs, 1 job on each node.
# Example usage: sbatch --array=0-4 runners/reader_sbatch.sh
# To launch all of the jobs from karray but only run 2 at a time:
# Example usage: sbatch --array=0-7%2 runners/reader_sbatch.sh
###


source setup/set_paths.sh
set -x

karray=(1 2 3 4 5 10 20 100)
k=${karray[$SLURM_ARRAY_TASK_ID]}

python3 reader/run.py \
    --config nq_mistral-7b-instruct_shot2_colbert_default.yaml \
    --prompt_file nq_default.json \
    --shot 0 \
    --eval_file nq_retrieval-colbert.json \
    --ndoc ${k}

#if [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
#    python3 reader/run.py \
#        --config nq_llama-2-7b-chat_shot1_gold_default.yaml \
#        --prompt_file nq_default.json \
#        --shot 0 \
#        --eval_file nq_gold.json \
#        --ndoc 5
#fi
