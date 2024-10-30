import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
convert dpr wiki split (used by alce as docs) to ColBERT format: .tsv with id and passage (no header)
"""
DATASET_PATH = os.environ.get("DATASET_PATH")


def main():
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
    main()