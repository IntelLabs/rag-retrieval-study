import argparse
import numpy as np
import os
from tqdm import tqdm
from file_utils import load_json, save_json
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#'asqa-gpt-35-turbo-gtr-shot2-ndoc5-42-azure.json', 'asqa-gpt-35-turbo-gtr_interact_search_summary-shot2-ndoc10-42-azure.json', 'asqa-vicuna-13b-rigorous-shot2-ndoc3-42.json'
file_path = '/home/vyvo/rag-svs/alce/results/workshop_data/human_eval_citations_completed.json'
human_cond = 'asqa-gpt-35-turbo-gtr-shot2-ndoc5-42-azure.json'
ret_path = '/export/data/vyvo/ALCE-data/asqa_eval_gtr_top100.json'
nret_docs = 5

output_path = f'/home/vyvo/rag-svs/alce/results/workshop_data/human-eval_{human_cond}'
gold_path = '/export/data/aleto/rag_data/asqa_eval_gold.json'
print(f"Adding {gold_path} data to {human_cond} in {file_path}")

# Load gold data
gold = load_json(gold_path)
all_gold_ids = np.array([gd['sample_id'] for gd in gold])
# Load retriever data
retd = load_json(ret_path)
all_ret_ids = np.array([rd['sample_id'] for rd in retd]) 

# Load human data + minor reformatting
hdata = load_json(file_path)
tmp = hdata['asqa'][human_cond]
data = [v for k, v in tmp.items() if k[1].isnumeric()]

# Loop through human data
for d in data:
    # Get data from gold
    g_ind = np.where(d['id'] == all_gold_ids)[0]
    assert len(g_ind) == 1, "Unexpected gold ID match!"
    g_ind = g_ind[0]
    query_qa = gold[g_ind]['qa_pairs']
    gold_docs = gold[g_ind]['docs']
    gold_titles = [gd['title'] for gd in gold_docs]
    gold_ids = [gd['id'] for gd in gold_docs]
    to_add = {}
    to_add["title_set"] = list(set(gold_titles)) # remove duplicates
    to_add["id_set"] = gold_ids
    # Get data from retriever
    r_ind = np.where(d['id'] == all_ret_ids)[0]
    assert len(r_ind) == 1, "Unexpected retriever ID match!"
    r_ind = r_ind[0]
    ret_docs = retd[r_ind]['docs'][:nret_docs]
    # Actually add to current data
    d['qa_pairs'] = query_qa
    d['gold_docs'] = to_add
    d['docs'] = ret_docs

print(f"Writing to {output_path}")
save_json(data, output_path)
