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
Samples 100 asqa queries for testing retrieval 
"""

def main():
    DATA_PATH = os.environ.get("DATA_PATH")
    input_path = os.path.join(
        DATA_PATH, 
        "asqa_eval.json"
    )

    input_data = load_json(input_path, logger)


    output_data = []
    logger.info("Processing data")
    for cur in tqdm(input_data[100:]):
        output_data.append(cur)
        
    output_path = os.path.join(
        DATA_PATH, 
        "asqa_eval-test.json"
    )
    save_json(output_data, output_path, logger)  # overwrite old file

if __name__ == "__main__":
    main()
