from file_utils import load_json
import convert_alce_utils

import os
import pandas as pd
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATASET_PATH = os.environ.get("DATASET_PATH")


def main(args):
    """
    set up alce data (qampari/asqa with DPR wiki docs) for ColBERT retrieval:
        1. convert alce queries to ColBERT format: .tsv with id and passage (no header)
        2. convert dpr wiki split (used by alce as docs) to ColBERT format: .tsv with id and passage (no header)
    """

    # convert alce queries to ColBERT format
    input_file = os.path.join(
        DATASET_PATH,
        args.dataset,
        "raw.json"
    )
    input_dict = load_json(input_file, logger)
    output_dict = convert_alce_utils.gen_colbert_queries(input_dict, args.dataset, logger)

    output_file = os.path.join(
        DATASET_PATH,
        args.dataset,
        "queries.tsv"
    )

    logger.info(f"Converting dictionary to pandas df")
    df = pd.DataFrame.from_dict(output_dict, orient='columns')

    logger.info(f"Writing output file {output_file}")
    df.to_csv(output_file, sep="\t", header=False, index=False) 

    # convert_dpr_wiki_to_colbert
    input_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "raw.tsv"  # dpr wiki split used by alce as docs
    )
    logger.info(f"Reading input file {input_file}")
    df = pd.read_csv(input_file, sep='\t')

    output_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "docs.tsv"
    )
    logger.info(f"Writing output file {output_file}")
    df.to_csv(output_file, sep="\t", header=False, index=False) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="ALCE dataset to compile eval files for: [asqa, qampari]")
    args = parser.parse_args()
    main(args)