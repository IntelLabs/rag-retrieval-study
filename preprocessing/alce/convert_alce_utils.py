from tqdm import tqdm
import pandas as pd

"""
Utility functions for converting ALCE data files
"""


def get_blank_eval(input_list, logger=None):
    """
    for removing 'docs' key from original alce data files (quampari and asqa)
    may be used to add custom retrieval results to the original alce eval files
    """
    if logger:
        logger.info("Removing retrieved docs from original ALCE eval file...")

    output_list = []
    for d in input_list:
        del d["docs"]
        output_list.append(d)
    return output_list


def fix_asqa_id(input_list, logger=None):
    """
    change "sample_id" key to "id" in alce-provided asqa eval files for consistency with other eval files
    """
    if logger:
        logger.info("Changing 'sample_id' key to 'id'...")

    clean_data = []
    for d in tqdm(input_list):
        d["id"] = d["sample_id"]
        del d["sample_id"]
        clean_data.append(d)
    return clean_data


def add_gold_to_eval(input_list, gold_data, logger=None):
    """
    Adds gold info to asqa and qampari alce eval file for evaluating retriever performance
    """
    if logger:
        logger.info("Adding gold data to eval files...")

    output_data = []
    for i, cur in tqdm(enumerate(input_list)):
        gold_docs = gold_data[i]['docs']
        gold_titles = [d["title"] for d in gold_docs]
        gold_ids = [str(d["id"]) for d in gold_docs]

        to_add = {}
        to_add["title_set"] = list(set(gold_titles)) # remove duplicates
        to_add["id_set"] = gold_ids

        cur["output"] = to_add
        cur['answer'] = gold_data[i]['answer']
        output_data.append(cur)


def gen_dpr_id2title(input_file, logger=None):
    """
    creates id2title.jsonl document for dpr_wiki
    allows id to be matched to title after svs search 
    """
    df = pd.read_csv(input_file, sep='\t')
    if logger:
        logger.info("Processing raw tsv...")
    output_data = []
    for _, row in tqdm(df.iterrows()):
        curr_dict = {}
        curr_dict["id"] = int(row["id"])
        curr_dict["title"] = str(row["title"])
        output_data.append(curr_dict)
    return output_data
