import os
import argparse
from tqdm import tqdm
from file_utils import load_json, save_json
import numpy as np
import copy
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Creates new eval files that achieve X% recall of the gold documents (rather than Y% search recall of the retriever nearest neighbors). Does this by simply calculating the exact number of gold documents to replace across all queries, then sampling that number for the whole dataset. This sampling ignores the query identity, meaning that each query will get a variable number of replaced documents.
"""
def get_last_n(lst, n):
    yield lst[len(lst)-n:len(lst)]

def main(args):
    DATA_PATH = os.environ.get("DATA_PATH")
    retrieved_results_file = os.path.join(
        DATA_PATH, 
        args.f
    )
    retrieved_results = load_json(retrieved_results_file, logger)

    gold_file = os.path.join(
        DATA_PATH, 
        args.gold_file
    )
    gold_docs = load_json(gold_file, logger)
    gold_replaced = copy.deepcopy(gold_docs)
   
    if 'asqa' in args.d:
        print("Adding 'output' field with gold docs for easy eval") 
        for i, cur in enumerate(gold_replaced):
            gold_titles = [d["title"] for d in gold_docs[i]['docs']]
            gold_ids = [str(d["id"]) for d in gold_docs[i]['docs']]
            to_add = {}
            to_add["title_set"] = list(set(gold_titles)) # remove duplicates
            to_add["id_set"] = gold_ids
            cur["output"] = to_add

    assert len(gold_docs) == len(retrieved_results), f"Should have the same number of gold eval queries ({len(gold_docs)}) as retrieved eval file ({len(retrieved_results)})"

    if 'nq' in args.d:   
        par_level = True
        # Figure out number of gold documents for each query 
        ngold = np.array([len(g['output']['page_par_id_set']) for g in gold_docs])
        total_gold = ngold.sum()
        g_bins = np.arange(1, 8)
        gold_ct, _ = np.histogram(ngold, bins=g_bins)
        query_idx_to_replace = []
        for gct, gi in zip(gold_ct, g_bins):
            # Array of indices to sample
            if gi < g_bins[-2]:
                print(f"Getting substitutes for queries with {gi} gold docs")
                query_inds = np.where(ngold == gi)[0]
                ng_per_bin = np.tile(np.arange(gct), gi)
            else:
                print(f"Getting substitutes for queries with {gi}+ gold docs")
                # For the remaining queries with >= 6 gold documents
                query_inds = np.where(ngold >= gi)[0]
                ng_per_bin = np.concatenate([np.ones(ngold[queryind], dtype=int) * ii for ii, queryind in enumerate(query_inds)])
            assert ng_per_bin.size == ngold[query_inds].sum(), "Unexpected number of gold documents for last bin!"
            # Figure out which queries/docs need to be replaced by a non-gold
            nsub = int(np.round(ng_per_bin.size * (1. - args.recall)))
            sub_inds = np.random.choice(ng_per_bin, nsub, replace=False)
            replace = query_inds[sub_inds]
            # A query index can appear more than once if it has multiple gold documents
            query_idx_to_replace.append(replace)
        all_idx_to_replace = np.concatenate(query_idx_to_replace)
    else:
        par_level = False
        assert args.d == 'asqa', "Dataset must be NQ or ASQA"
        # Always have 5 gold documents in ASQA
        ngold = np.ones(len(gold_docs), dtype=int) * 5
        goldq_arr = np.tile(np.arange(len(gold_docs)), 5)
        total_gold = goldq_arr.size
        nsub = int(np.round(total_gold * (1. - args.recall)))
        all_idx_to_replace = np.random.choice(goldq_arr, nsub, replace=False)
    print(f"Replacing {all_idx_to_replace.size} gold documents from {total_gold} total across {len(gold_docs)} queries in {args.d}")
    uniq_idx, uniq_cts = np.unique(all_idx_to_replace, return_counts=True)
    #print(f"Replacing {uniq_idx}")

    # Iterate through queries to replace
    for qi, nreplace in zip(uniq_idx, uniq_cts):
        this_gold = gold_docs[qi]['docs']
        try:
            assert gold_docs[qi]['id'] == retrieved_results[qi]['id'], f"Misalignment between gold and retrieved query results!"
        except:
            assert gold_docs[qi]['sample_id'] == retrieved_results[qi]['id'], f"Misalignment between gold and retrieved query results!"
        gold_ids = [int(g['id']) for g in this_gold]
        subi = np.random.choice(ngold[qi], nreplace, replace=False)
        this_ret = iter(retrieved_results[qi]['docs'])
        replace_docs = []
        while len(replace_docs) < nreplace:
            x = next(this_ret)
            if int(x['id']) not in gold_ids:
                replace_docs.append(x)
        #### debug
        #replace_ids = [r['id'] for r in replace_docs]
        #print(f"Replacing inds {subi} of gold ids {gold_ids} with {replace_ids}")
        #import pdb; pdb.set_trace()
        #### debug
        for ii, doc in zip(subi, replace_docs):
            this_gold[ii] = doc
        gold_replaced[qi]['docs'] = this_gold

    # write files
    out_dir = os.path.join(DATA_PATH, 'gold_recall')
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{args.d}_gold_recall{args.recall}.json")
    save_json(gold_replaced, output_file)
    print(f"Saved to {output_file}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Name of file in $DATA_DIR with desired retrieval results")
    parser.add_argument("--gold_file", type=str, help="Name of file in $DATA_DIR with gold data")
    parser.add_argument("--d", type=str, help="Name of dataset")
    parser.add_argument("--recall", type=float, help="Gold recall (between 0 and 1)")

    args = parser.parse_args()
    main(args)
