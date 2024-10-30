from file_utils import load_json, save_json, save_jsonl
import convert_alce_utils

import os
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATASET_PATH = os.environ.get("DATASET_PATH")
DATA_PATH = os.environ.get("DATA_PATH")


def main(args):
    """
    convert ALCE dataset eval files (asqa, qampari) to format used for retrieval and retrieval eval
    creates id2title.jsonl for svs retrieval
    """

    # load raw eval
    input_file = os.path.join(
        DATASET_PATH,
        args.dataset,
        "raw.json"  # original .json alce eval file
    )
    input_eval = load_json(input_file, logger)

    # load gold data
    gold_path = os.path.join(
        DATA_PATH,
        f"{(args.dataset).lower()}_gold.json"
    )
    gold_data = load_json(gold_path, logger)

    # remove 'docs' key from original alce data files (quampari and asqa)
    input_eval = convert_alce_utils.get_blank_eval(input_eval, logger)

    # convert 'sample_id' to 'id' in asqa eval files
    input_eval = convert_alce_utils.fix_asqa_id(input_eval, logger)

    # add gold info to asqa and qampari alce eval file for evaluating retriever performance
    output_eval = convert_alce_utils.add_gold_to_eval(input_eval, gold_data, logger)

    # save modified data
    output_file = os.path.join(
        DATA_PATH,
        f"{args.dataset}_eval.json"
    )
    save_json(output_eval, output_file, logger)

    # load raw dpr wiki data
    dpr_input_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "raw.tsv"
    )
    logger.info(f"Reading input file {input_file}")

    # convert dpr wiki split (used by alce as docs) to format used to generate vectors for svs
    output_file = convert_alce_utils.gen_dpr_wiki_jsonl(dpr_input_file, logger)
    output_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "docs.jsonl"
    )
    save_jsonl(output_file, output_file, logger)

    # generate dpr id2title json for svs retrieval with qampari and asqa
    dpr_id2title = convert_alce_utils.gen_dpr_id2title(dpr_input_file)

    output_file = os.path.join(
        DATASET_PATH,
        "dpr_wiki",
        "id2title.jsonl"
    )
    save_jsonl(dpr_id2title, output_file, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="ALCE dataset to compile eval files for: [asqa, qampari]")
    args = parser.parse_args()
    main(args)