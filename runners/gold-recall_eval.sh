#!/bin/bash

source setup/set_paths.sh
set -x

recallarray=(0.5 0.7 0.9)
model_id="Mistral-7B-Instruct-v0.3" #"Llama-2-7b-chat-hf"
nshot=2 #1

for recall in "${recallarray[@]}"; do
    # NQ with citations
    evalfile=$(ls results/reader/gold-recall/nq-${model_id}-None-shot${nshot}*-42-cite-bge-base-gold-recall-${recall}.json | xargs -n 1 basename)
    python -u reader/eval_per_query.py --citations --f gold-recall/${evalfile}

    # NQ without citations
    evalfile=$(ls results/reader/gold-recall/nq-${model_id}-None-shot0-*-42-bge-base-gold-recall-${recall}.json | xargs -n 1 basename)
    python -u reader/eval_per_query.py --f gold-recall/${evalfile}
    
    # ASQA with citations
    evalfile=$(ls results/reader/gold-recall/asqa-${model_id}-None-shot${nshot}*-42-cite-bge-base-gold-recall-${recall}.json | xargs -n 1 basename)
    python -u reader/eval_per_query.py --citations --f gold-recall/${evalfile}
done
