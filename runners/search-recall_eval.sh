#!/bin/bash

source setup/set_paths.sh
set -x

recallarray=(0.7 0.9 0.95)
model_id="Llama-2-7b-chat-hf" #"Mistral-7B-Instruct-v0.3"
nshot=1  #2

for recall in "${recallarray[@]}"; do
    # NQ with citations
    evalfile=$(ls results/reader/search-recall/nq-${model_id}-None-shot${nshot}*-42-cite-bge-base-search-recall-${recall}.json | xargs -n 1 basename)
    python -u reader/eval_per_query.py --citations --f search-recall/${evalfile}

    # NQ without citations
    evalfile=$(ls results/reader/search-recall/nq-${model_id}-None-shot0-*-42-bge-base-search-recall-${recall}.json | xargs -n 1 basename)
    python -u reader/eval_per_query.py --f search-recall/${evalfile}
    
    # ASQA with citations
    evalfile=$(ls results/reader/search-recall/asqa-${model_id}-None-shot${nshot}*-42-cite-bge-base-search-recall-${recall}.json | xargs -n 1 basename)
    python -u reader/eval_per_query.py --citations --f search-recall/${evalfile}
done
