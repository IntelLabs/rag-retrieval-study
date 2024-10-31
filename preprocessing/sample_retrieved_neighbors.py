import os
import argparse
from tqdm import tqdm
from file_utils import load_json, save_json
import pandas as pd
import copy
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Creates new eval files for retrieved results 5-10 and 95-100 for testing bin experiment
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

    # remove gold documents from retrieved results, concat to 100
    logger.info("Removing gold documents and 5 top retrieved results, concat to len(100)...")
    for item in tqdm(retrieved_results):
        par_level = False
        try: 
            gold_ids = item['output']['id_set']
        except: 
            par_level = True
            gold_ids = item['output']['page_par_id_set']
        clean_docs = []
        # first 5 removed, use as 'retrieved'
        for d in item['docs'][5:]:
            cur_id = d['id']
            if par_level:
                cur_id = cur_id + '_' + d['start_par_id']
            if cur_id not in gold_ids:
                clean_docs.append(d)
        item['docs'] = clean_docs[:100]

    bin_list = []
    logger.info("Creating")

    # Create "bin0" which has nearest neighbors 5-10
    bin = []
    for item in tqdm(retrieved_results):
        item_copy = copy.deepcopy(item)
        item_copy['docs'] = item_copy['docs'][:5]
        bin.append(item_copy)
    bin_list.append(bin)

    # Create "bin1" which has nearest neighbors 95-100
    bin = []
    for item in tqdm(retrieved_results):
        item_copy = copy.deepcopy(item)
        item_copy['docs'] = item_copy['docs'][-5:]
        bin.append(item_copy)
    bin_list.append(bin)

    # write files
    out_dir = os.path.join(DATA_PATH, 'noise_bins')
    os.makedirs(out_dir, exist_ok=True)
    
    for i, bin in enumerate(bin_list):
        save_json(bin, os.path.join(out_dir, f"{args.d}_bin{i}.json") )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Name of file in $DATA_DIR with desired retrieval results")
    parser.add_argument("--d", type=str, help="Name of dataset")

    args = parser.parse_args()
    main(args)
