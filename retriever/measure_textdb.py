import argparse
import json
import os
import logging
import yaml
import numpy as np
import torch
from tqdm import tqdm
import pysvs

# import indexing functions for all retrievers
import retriever.index as index
from utils import InvalidArgument, IncompleteSetup
from file_utils import save_json, load_json, save_pickle, load_pickle

DATA_PATH = os.environ.get("DATA_PATH")
DATASET_PATH = os.environ.get("DATASET_PATH")
COLBERT_MODEL_PATH = os.environ.get("COLBERT_MODEL_PATH")
INDEX_PATH = os.environ.get("INDEX_PATH")
VEC_PATH = os.environ.get("VEC_PATH")


def dense_retrieval(
    query_data: dict,
    k: int,
    doc_dataset: str,
    embed_file: str,
    embed_model_name: str,
    embed_model_type: str,
    index_fn,
    dist_type,
    batch_size: int,
    text_key: str,
    index_kwargs: dict,
    logger
):
   
    """
    Dense SVS retrieval assuming the corpus of vector embeddings have already been created
    """
    import gc

    index_path = os.path.join(INDEX_PATH, 'dense', embed_file.split(".fvecs")[0])
    vec_file = os.path.join(VEC_PATH, embed_file)
    
    # Save direct search outputs before writing to JSON
    #save_pickle([k_neighbors, dist_neighbors], f'{doc_dataset}_tmp.pkl', logger)
    queries = [d["question"] for d in query_data]  # change to question
    queries = queries[:batch_size*4]  # DEBUG
    k_neighbors, dist_neighbors = load_pickle(f'{doc_dataset}_tmp0.pkl', logger)

    # Default empty data dict if neighbor is not found in dataset
    empty_dict = {
        "id": "",
        "start_par_id": "",
        "end_par_id": "",
        "title": "",
        "text": "",
        "score": ""
    }
    import time
    import datasets

    t0 = time.time()
    datasets.config.IN_MEMORY_MAX_SIZE = 600 * 1e9  # allow large datasets to be entirely held in memory
    # Load in the corpus to get the text associated with the retrieved documents
    corpus_file = os.path.join(DATASET_PATH, doc_dataset, "docs.jsonl")
    corpus = datasets.load_dataset("json", data_files=corpus_file)
    corpus = corpus['train']
    # Load in the titles of the documents
    doc_titles = datasets.load_dataset("json", data_files=os.path.join(DATASET_PATH, doc_dataset, "id2title.jsonl"))
    doc_titles = doc_titles["train"].to_dict()
    title_ids, titles = doc_titles['id'], doc_titles['title']
    title_dict = {str(id): title[0] for id, title in zip(title_ids, titles) if title is not None}
    del doc_titles

    for qi, q in enumerate(tqdm(queries)):
        neighbor_inds = k_neighbors[qi, :]
        neighbor_data = corpus[neighbor_inds]
        # Get associated text
        n_text = neighbor_data[text_key]
        # Get document & passage ID. Also get associated document title.
        n_id = neighbor_data["id"]
        tmp = np.array([x.split("_") for x in n_id])
        docs = tmp[:, 0]
        paras = tmp[:, 1]
        # titles = doc_titles.filter(lambda x: np.isin(x['id'], docs), batched=True, batch_size=100).to_dict()
        # title_dict = {id: title for id, title in zip(title_ids, titles)}
        ret = \
        [
            {
                "id": doc_id,
                "start_par_id": par_id,
                "end_par_id": par_id,
                "title": title_dict[doc_id],
                "text": txt,
                "score": str(dist_neighbors[qi, j])

            } for j, (doc_id, par_id, txt) in enumerate(zip(docs, paras, n_text))
            #if neighbor_data[j] else empty_dict for j in range(k)
        ]
        query_data[qi]['docs'] = ret
    logger.info(f"First method took {time.time() - t0}")


    t1 = time.time()
    corpus_file = os.path.join(DATASET_PATH, doc_dataset, "docs.jsonl")
    with open(corpus_file, 'r') as f:
        clines = f.readlines()
    #
    # empty_dict = {
    #     "id": "",
    #     "start_par_id": "",
    #     "end_par_id": "",
    #     "title": "",
    #     "text": "",
    #     "score": ""
    # }
    for qi, q in enumerate(tqdm(queries)):
        neighbor_ids = k_neighbors[qi, :]
        try:
            # read lines from json corresponding to neighbors
            wp_dicts = [json.loads(clines[int(n)]) if n != 0 else None for n in neighbor_ids]
        except Exception as e:
            logger.info(e)
            import pdb; pdb.set_trace()
        # loop over neighbor dictionaries (pulled from json)
        ret = \
        [
            {
                "id": wp_dicts[j]["id"].split("_")[0],
                "start_par_id": wp_dicts[j]["id"].split("_")[1],
                "end_par_id": wp_dicts[j]["id"].split("_")[1],
                "title": None,  #fix this?
                "text": wp_dicts[j][text_key],
                "score": str(dist_neighbors[qi, j])

            } if wp_dicts[j] else empty_dict for j in range(k)
        ]
        query_data[qi]['docs'] = ret
    logger.info(f"Second method took {time.time() - t1}")

    logger.info('DONE')
    return query_data


def main(args): 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # check that index file path env variable has been set 
    if not os.path.exists(INDEX_PATH):
        logger.info("Please set environment variable $INDEX_PATH to specify where to save or load indices.")
        raise IncompleteSetup


    # check that query file exists
    query_file = os.path.join(DATA_PATH, args.query_file)
    if not os.path.exists(query_file):
        logger.info(f"Query file specified does not exist: {query_file}")
        raise InvalidArgument
    else:
        # open specified data file (containing queries)
        # will append retrieved docs to this file
        query_data = load_json(query_file, logger)

    if args.retriever == "colbert":
        data = colbert_retrieval(
            query_data,
            args.k,
            args.query_dataset,
            args.doc_dataset,
            logger
        )
    elif args.retriever == "dense":
         # check that all required arguments are specified
        if not args.embed_file:
            logger.info("path to .fvecs vector embedding file from $VEC_PATH must be specified for retrieval with svs")
            raise InvalidArgument
        if not args.embed_model_name:
            logger.info("model to use for embedding queries with SentenceTransformer (eg. 'snowflake/snowflake-arctic-embed-s') must be specified for retrieval with svs")
            raise InvalidArgument

        data = dense_retrieval(
            query_data,
            args.k,
            args.doc_dataset,
            args.embed_file,
            args.embed_model_name,
            args.embed_model_type,
            args.index_fn,
            args.dist_type,
            args.batch_size,
            args.text_key,
            args.index_kwargs,
            logger
        )
    else:
        print(f"Invalid retriever: {args.retriever}")
        print("Current implemented options include: colbert/dense")
        raise InvalidArgument

    # if output_file specified, create new .json in $DATA_PATH
    # otherwise, add neighbors to input file
    if args.output_file:
        output_file_name = args.output_file
        if not output_file_name.endswith('.json'):
            output_file_name += '.json'
        output_file = os.path.join(DATA_PATH, args.output_file)
    else:
        output_file = query_file
    save_json(data, output_file, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--config", type=str, default=None, help="name of config file in retriever/configs/")
    parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--retriever", type=str, help="options: colbert/dense")
    parser.add_argument("--query_file", type=str, help="path to the data file with queries")
    parser.add_argument("--query_dataset", type=str, help="query dataset name -> options: nq")
    parser.add_argument("--doc_dataset", type=str, help="dataset name -> options: kilt_wikipedia/// ")
    parser.add_argument("--output_file", type=str, default=None, help="Same format as the data file but with the retrieved docs; if not specified, will be added to the original data_file")
 
    # svs arguments
    parser.add_argument("--embed_file", default=None, help='path to .fvecs vector embedding file from $VEC_PATH')
    parser.add_argument("--embed_model_name", default=None, help='model to use for embedding queries with SentenceTransformer (eg. "snowflake/snowflake-arctic-embed-s")')
    parser.add_argument("--embed_model_type", default="st", help="Type of embedding model to use, choose from [st, hf]. st is SentenceTransformers and hf is HuggingFace")
    parser.add_argument("--index_fn", default=pysvs.Flat, help='type of pysvs index used for dense retrieval')
    parser.add_argument("--dist_type", default=pysvs.DistanceType.MIP, help='type of distance to use for index/search')
    parser.add_argument("--batch_size", type=int, default=32, help='query batch size')
    parser.add_argument("--text_key", type=str, default="text", help='key in data dictionary')
    parser.add_argument("--index_kwargs", type=json.loads, default="{}", help='additional input arguments for index building')
    args = parser.parse_args()

    # get defaults from config file
    if args.config:
        config = os.path.join(os.getcwd(), 'retriever', 'configs', args.config)
        config = yaml.safe_load(open(config))
        parser.set_defaults(**config)

    args = parser.parse_args()

    main(args)

