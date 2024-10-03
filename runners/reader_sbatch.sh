#!/bin/bash
#SBATCH --job-name=reader_k
#SBATCH --partition=gpu-p
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 3-0


###
# The following command would launch 5 jobs, 1 job on each node.
# Example usage: sbatch --array=0-4 runners/reader_sbatch.sh
# To launch all of the jobs from karray but only run 2 at a time:
# Example usage: sbatch --array=0-7%2 runners/reader_sbatch.sh
###


source setup/set_paths.sh
set -x

karray=(1 2 3 4 5 10 20)
k=${karray[$SLURM_ARRAY_TASK_ID]}

python3 reader/run.py \
    --config bioasq_mistral-7b-instruct_shot0_ndoc5_svs_default.yaml \
    --eval_file bioasq_retrieval-bge-base-dense.json \
    --ndoc ${k} \
&>> logs/log_reader_bioasq_bge-base_k${k}.txt

