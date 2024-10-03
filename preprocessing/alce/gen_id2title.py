import os
import json
from tqdm import tqdm
from file_utils import save_jsonl
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
creates id2title.jsonl document for dpr_wiki
allows id to be matched to title after svs search 
"""

def main():
    DATASET_PATH = os.environ.get("DATASET_PATH")
    input_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "raw.tsv"
    )
    logger.info(f"Reading input file {input_file}")
    df = pd.read_csv(input_file, sep='\t')

    output_data = []
    logger.info("Iterating over raw tsv")
    for _, row in tqdm(df.iterrows()):
        curr_dict = {}
        curr_dict["id"] = int(row["id"])
        curr_dict["title"] = str(row["title"])
        output_data.append(curr_dict)
    
    output_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "id2title.jsonl"
    )
    
    save_jsonl(output_data, output_file, logger)
        




if __name__ == "__main__": 
    main()