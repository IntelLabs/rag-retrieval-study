import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from file_utils import save_jsonl

"""
convert dpr wiki split (used by alce as docs) to format used to generate vectors
"""
DATASET_PATH = os.environ.get("DATASET_PATH")

def main():
    input_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "raw.tsv"
    )
    logger.info(f"Reading input file {input_file}")
    df = pd.read_csv(input_file, sep='\t')

    output_data = []
    for ind in df.index:
        doc_id = df['id'][ind]
        doc_text = df['text'][ind]

        output_dict = {}
        output_dict['id'] = int(doc_id)
        output_dict['text'] = str(doc_text)
        output_data.append(output_dict)

    output_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "docs.jsonl"
    )
    save_jsonl(output_data, output_file, logger)


if __name__ == "__main__": 
    main()