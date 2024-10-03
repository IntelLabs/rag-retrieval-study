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

def colbert_retrieval(
    query_data,
    k: int,
    query_dataset,
    doc_dataset,
    logger
):
    """
    Index (or load existing index) and retrieve with ColBERT
    """
    from retriever.colbert.data import Queries
    from retriever.colbert.infra import Run, RunConfig, ColBERTConfig
    from retriever.colbert import Searcher


    model_path = os.path.join(  # path to model used for indexing
        COLBERT_MODEL_PATH,
        'colbertv2.0/'
    )
    query_path = os.path.join(  # path to input query tsv 
        DATASET_PATH,
        query_dataset,
        "queries.tsv"
    )
    doc_path = os.path.join(
        DATASET_PATH,
        doc_dataset,
        "docs.tsv"
    )
    index_ret_path = os.path.join(
        INDEX_PATH,
        'colbert',
        doc_dataset
    )

    exp_name = f'colbert'
    prediction_temp_dir = ""  # TODO

    with Run().context(
        RunConfig(
            nranks=1,
            experiment=exp_name,
            index_root=index_ret_path,
        )
    ):
        # index documents in $INDEX_DIR/colbert/doc_dataset
        index_ds_path = os.path.join(index_ret_path, doc_dataset)
        config = ColBERTConfig(
            index_path=index_ds_path,
            nbits=2,
            root=prediction_temp_dir,
        )
        logger.info(f"Config for Colbert retrieval: {config}")
        
    
        logger.info("Start indexing....")
        index.colbert_build_index(
            config,
            model_path,
            doc_dataset,
            doc_path,
            logger
        )
        logger.info("Indexing complete")

        searcher = Searcher(index=doc_dataset, config=config)
        
        logger.info(f"Loading queries from: {query_path}")
        queries = Queries(query_path)

        logger.info("Starting search...")
        ranking = searcher.search_all(queries, k=100)

        for r_id, r in enumerate(ranking.data):
            ret = []
            for i, (c_id, rank, score) in enumerate(ranking.data[r]):
                page_id, paragraph_id = searcher.collection.pid_list[c_id].split('_')
                ret.append({
                    "id": page_id,
                    "start_par_id": paragraph_id,
                    "end_par_id": paragraph_id,
                    "title": None,  # need this?
                    "text": searcher.collection.data[c_id],
                    "score": score
                })

            query_data[r]["docs"] = ret  # double check I'm putting this in the right spot
            q = q.to("cpu")

        logger.info("Search complete")

    return query_data

def dense_retrieval(
    query_data: dict,
    k: int,
    doc_dataset: str,
    embed_file: str,
    embed_model_name: str,
    embed_model_type: str,
    index_fn,
    dist_type,
    index_kwargs,
    logger
):
   
    """
    Dense SVS retrieval assuming the corpus of vector embeddings have already been created
    """
    import gc
    import pickle

    index_path = os.path.join(INDEX_PATH, 'dense', embed_file.split(".fvecs")[0])
    vec_file = os.path.join(VEC_PATH, embed_file)

    logger.info('Start indexing...')
    i = index.dense_build_index(
        index_path,
        vec_file,
        index_fn,
        dist_type,
        index_kwargs,
        logger
    )
    logger.info('Done indexing')

    logger.info('Embedding queries...')
    queries = [d["question"] for d in query_data]  # change to question
    if embed_model_type == 'st':
        import sentence_transformers as st
        embed_model = st.SentenceTransformer(embed_model_name)
        query_embs = embed_model.encode(queries)
    else:
        raise NotImplementedError

    logger.info('Start searching...')
    k_neighbors, dist_neighbors = i.search(query_embs, args.k)
    logger.info('Done searching')

    # Save direct search outputs before writing to JSON
    save_pickle([k_neighbors, dist_neighbors], 'tmp.pkl', logger)
    del i
    gc.collect() 

    # # open temp pickle
    # [k_neighbors, dist_neighbors] = load_pickle('tmp.pkl', logger)
    
    corpus_file = os.path.join(DATASET_PATH, doc_dataset, "docs.jsonl")
    with open(corpus_file, 'r') as f:
        clines = f.readlines()

    empty_dict = {
        "id": "",
        "start_par_id": "",
        "end_par_id": "",
        "title": "",
        "text": "",
        "score": ""
    }
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
                "text": wp_dicts[j]["text"],
                "score": str(dist_neighbors[qi, j])

            } if wp_dicts[j] else empty_dict for j in range(k)
        ]
        query_data[qi]['docs'] = ret

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
    parser.add_argument("--index_kwargs", type=json.loads, default="{}", help='additional input arguments for index building')
    args = parser.parse_args()

    # get defaults from config file
    if args.config:
        config = os.path.join('retriever', 'configs', args.config)
        config = yaml.safe_load(open(config))
        parser.set_defaults(**config)

    args = parser.parse_args()

    main(args)
