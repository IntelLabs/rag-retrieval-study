import os
import pandas as pd
import argparse
from file_utils import load_json, save_json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
script for removing 'docs' key from original alce data files (quampari and asqa)
"""
DATASET_PATH = os.environ.get("DATASET_PATH")
DATA_PATH = os.environ.get("DATA_PATH")

def main(args):
    input_file = os.path.join(
        DATASET_PATH,
        args.dataset,
        "raw.json"
    )
    input_list = load_json(input_file, logger)
    output_list = []
    for d in input_list:
        del d["docs"]
        output_list.append(d)

    output_file = os.path.join(
        DATA_PATH,
        f"{args.dataset}_eval.json"  # "qampari_eval.json",
    )
    save_json(output_list, output_file, logger)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to remove docs key from")
    args = parser.parse_args()
    main(args)