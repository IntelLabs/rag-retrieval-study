import os
from tqdm import tqdm
import argparse
from file_utils import load_json, save_json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Change "sample_id" key to "id" in alce-provided asqa eval files for consistency with other eval files
"""


def main(args):

    DATA_PATH = os.environ.get("DATA_PATH")
    input_file = os.path.join(DATA_PATH, args.eval_file)

    # adds gold answer to nq eval file
    input_data = load_json(input_file)

    clean_data = []
    for d in tqdm(input_data):
        d["id"] = d["sample_id"]
        del d["sample_id"]
        clean_data.append(d)

    save_json(clean_data, input_file, logger)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--eval_file", help="Eval file name in $DATA_DIR to change id")
    args = parser.parse_args()
    main(args)