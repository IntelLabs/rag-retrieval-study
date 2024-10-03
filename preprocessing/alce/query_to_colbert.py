import os
import pandas as pd
from file_utils import load_json, save_json
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
convert alce queries to ColBERT format: .tsv with id and passage (no header)
"""
DATASET_PATH = os.environ.get("DATASET_PATH")

def main(args):
    input_file = os.path.join(
        DATASET_PATH,
        args.dataset,
        "raw.json"
    )
    input_dict = load_json(input_file, logger)
    output_dict = {}
    output_dict['id'] = []
    output_dict['query_text'] = []

    for entry in input_dict:
        if args.dataset == 'asqa':
            query_id = entry['sample_id']
        else: 
            query_id = entry['id']
        query_text = entry['question']
        output_dict['id'].append(query_id)
        output_dict['query_text'].append(query_text)

    output_file = os.path.join(
        DATASET_PATH,
        args.dataset,
        "queries.tsv"
    )
    
    logger.info(f"Converting dictionary to pandas df")
    df = pd.DataFrame.from_dict(output_dict, orient='columns')

    logger.info(f"Writing output file {output_file}")
    df.to_csv(output_file, sep="\t", header=False, index=False) 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to remove docs key from")
    args = parser.parse_args()
    main(args)