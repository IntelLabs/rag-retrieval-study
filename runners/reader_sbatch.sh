#!/bin/bash
#SBATCH --job-name=nocite-asqa
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
nk=${#karray[@]}

python3 reader/run.py \
    --config asqa_mistral-7b-instruct_shot0_bge_base_nocite.yaml \
    --ndoc ${k}

if [ $SLURM_ARRAY_TASK_ID -eq $nk ]; then
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot0_gold.yaml
elif [ $SLURM_ARRAY_TASK_ID -eq $((nk + 1)) ]; then
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot0_ndoc0_closedbook.yaml
fi

