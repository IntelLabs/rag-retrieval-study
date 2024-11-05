# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import os
from tqdm import tqdm
from utils import InvalidArgument, IncompleteSetup

from file_utils import save_json, load_json, save_pickle, load_pickle

DATA_PATH = os.environ.get("DATA_PATH")
DATASET_PATH = os.environ.get("DATASET_PATH")
COLBERT_MODEL_PATH = os.environ.get("COLBERT_MODEL_PATH")
INDEX_PATH = os.environ.get("INDEX_PATH")
VEC_PATH = os.environ.get("VEC_PATH")

def get_title_dict(doc_dataset):
    """
    uses datasets library to create dict where key is doc id and value is doc title
    """
    import datasets
    # Load in the titles of the documents
    doc_titles = datasets.load_dataset("json", data_files=os.path.join(DATASET_PATH, doc_dataset, "id2title.jsonl"))
    doc_titles = doc_titles["train"].to_dict()
    title_ids, titles = doc_titles['id'], doc_titles['title']
    title_dict = {str(id): title for id, title in zip(title_ids, titles)}
    return title_dict


def get_title(title_dict, doc_id, logger):
    """
    returns title given doc_id, outputs warning if none found
    """
    try:
        doc_id = str(doc_id)
        this_title = title_dict[doc_id]
    except Exception as e:
        logger.warning(f"No title found for {doc_id}, setting to blank")
        this_title = ""
    return this_title

def get_doc(doc_dataset, doc_id, doc_text, score, title_dict, logger):
    """
    get formatted dictionary for one neighbor doc (to append to "docs" list in output .json)
    """
    if doc_dataset == "dpr_wiki":
        this_title = get_title(title_dict, doc_id, logger)
        res_dict = {
            "id": str(doc_id),
            "title": this_title,
            "text": doc_text,
            "score": score
        }
    else:
        page_id, paragraph_id = doc_id.split('_')
        this_title = get_title(title_dict, page_id, logger)
        res_dict = {
            "id": page_id,
            "start_par_id": paragraph_id,
            "end_par_id": paragraph_id,
            "title": this_title,
            "text": doc_text,
            "score": score
        }
    
    return res_dict

def save_file(output_file_name, data, logger):
    """
    saves json output in $DATA_PATH
    """
    if not output_file_name.endswith('.json'):
        output_file_name += '.json'
    output_file = os.path.join(DATA_PATH, output_file_name)
    save_json(data, output_file, logger)


def save_noise(query_data, queries, k, k_neighbors, corpus, dist_neighbors, doc_dataset, text_key, title_dict, output_file_name, logger): 
    """
    Given neighbor dict from dense retrieval, writes output files for each "percentile" of neighbors
    random 10 docs in each percentile 
    """

    logger.info('Saving text and titles for each neighbor')

    import random
    random.seed(42)

    for i in range(10): # for each percentile
        logger.info(f"Generating {(i+1)*10}th percentile random noise")
        start_index = int(i * 0.01 * k)
        end_index = int((i + 1) * 0.01 * k)
        # for qi, q in enumerate(tqdm(queries)):
        for qi, q in enumerate(queries):
            # get gold ids for query
            par_gold = False
            try:
                gold_ids = query_data[qi]['output']['id_set']
            except:
                par_gold = True
                gold_ids = query_data[qi]['output']['page_par_id_set']
            # get neighbor info for this percentile only
            neighbor_inds = k_neighbors[qi, :]
            neighbor_inds = neighbor_inds[start_index:end_index]
            neighbor_data = corpus[neighbor_inds]
            # Get associated text
            n_text = neighbor_data[text_key]
            # Get document & passage ID. Also get associated document title.
            n_id = neighbor_data["id"]

            ret = []  # list of doc dicts
            choices = []  # track so no duplicates and no golds
            while len(choices) < 100:
                c = random.randrange(len(n_id))
                og_index = c + start_index
                doc_id = str(n_id[c])
                if doc_id not in choices and doc_id not in gold_ids:
                    # good choice!
                    choices.append(doc_id)
                    score = str(dist_neighbors[qi, og_index])
                    doc_text = n_text[c]
                    res_dict = get_doc(doc_dataset, doc_id, doc_text, score, title_dict, logger)
                    res_dict['neighbor_id'] = str(og_index)
                    ret.append(res_dict)
                # else continue generating
            query_data[qi]['docs'] = ret

        # output percentile json so memory isn't exceeded 
        percentile_file_name = output_file_name.split(".json")[0] + "-random-" + str((i+1)*10) + ".json"
        # load existing file if it exists
        if os.path.exists(os.path.join(DATA_PATH, percentile_file_name)):
            prev_batch_file = load_json(os.path.join(DATA_PATH, percentile_file_name))
            query_data = prev_batch_file + query_data
        save_file(percentile_file_name, query_data, logger)
