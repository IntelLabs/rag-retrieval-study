import os
import argparse
from tqdm import tqdm
from file_utils import load_json, save_json
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Adds gold info to asqa and qampari eval file for evaluating retriever performance
"""

def main(args):
    DATA_PATH = os.environ.get("DATA_PATH")
    input_path = os.path.join(
        DATA_PATH, 
        args.f
    )

    input_data = load_json(input_path, logger)

    gold_path = os.path.join(
        DATA_PATH, 
        f"{(args.dataset).lower()}_gold.json"
    )
    gold_data = load_json(gold_path, logger)

    # if len(input_data) != len(gold_data):
    #     raise Exception("len(input_data) != len(gold_data)")
    

    output_data = []
    logger.info("Processing data")
    for i, cur in tqdm(enumerate(input_data)):
        gold_docs = gold_data[i]['docs']
        gold_titles = [d["title"] for d in gold_docs]
        gold_ids = [str(d["id"]) for d in gold_docs]

        to_add = {}
        to_add["title_set"] = list(set(gold_titles)) # remove duplicates
        to_add["id_set"] = gold_ids

        cur["output"] = to_add
        cur['answer'] = gold_data[i]['answer']
        output_data.append(cur)

    save_json(output_data, input_path)  # overwrite old file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Name of file in $DATA_DIR to add gold data to")
    parser.add_argument("--dataset", type=str, help="Dataset whose gold data is being added: asqa/qampari")
    args = parser.parse_args()
    main(args)
