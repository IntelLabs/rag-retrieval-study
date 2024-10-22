#!/bin/bash


source setup/set_paths.sh

searchrec=(0.95 0.9 0.7)

for sr in "${searchrec[@]}"; do
    python -u retriever/eval.py --no_plot --ci --not_par_level --eval_file asqa_retrieval_ann-recall${sr}_bge-base-dense.json |& tee -a logs/ret_eval/log_goldrecforsearchrec_asqa.txt
    python -u retriever/eval.py --no_plot --ci --eval_file nq_retrieval_ann-recall${sr}_bge-base-dense.json |& tee -a logs/ret_eval/log_goldrecforsearchrec_nq.txt
done
