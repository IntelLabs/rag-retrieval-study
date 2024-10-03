#!/bin/bash
#SBATCH --job-name=reader_k
#SBATCH --partition=gpu-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -t 2-0

source setup/set_paths.sh
set -x

karray=(1 100)
k=${karray[$SLURM_ARRAY_TASK_ID]}

python3 reader/run.py \
    --config bioasq_mistral-7b-instruct_shot0_ndoc5_svs_default.yaml \
    --eval_file bioasq_eval-colbert.json \
    --ndoc ${k} \
&>> logs/log_reader_bioasq_colbert_k${k}.txt

<<<<<<< Updated upstream
for k in "${k_vals[@]}"; do
    python3 reader/run.py \
        --config asqa_mistral-7b-instruct_shot2_colbert_default.yaml \
        --ndoc ${k}
done
=======
#SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
#export PYTHONPATH=${SCRIPT_PATH}

#k_vals=("1" "2" "3" "4" "5" "10" "20" "100")
#k_vals=("2" "3" "4" "5" "10" "20" "100")
#
#
#for k in "${k_vals[@]}"; do
#    srun -p gpu-a --gres=gpu:2 --exclusive \
#        python3 reader/run.py \
#            --config bioasq_mistral-7b-instruct_shot0_ndoc5_svs_default.yaml \
#            --eval_file bioasq_eval-colbert.json \
#            --ndoc ${k} \
#    &> logs/log_reader_bioasq_colbert_k${k}.txt &
#    sleep 1m;
#done
>>>>>>> Stashed changes
