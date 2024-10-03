import argparse
import gc
import json
import os, sys
import logging
import yaml

import pysvs

# import indexing functions for all retrievers
from retriever.ret_utils import *
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
    output_file_name,
    logger
):
    """
    ColBERT retrieval indexing, searching, and output neighbors. Indexing already completed for doc_dataset will be reused
    It assumes these things exist:
        (1) Directory with downloaded colbert v2.0
        (2) tsv file containing query id and query text, named '{DATASET_PATH}/{dataset_name}/queries.tsv'
        (3) tsv file containing doc id, doc text, named '{DATASET_PATH}/{doc_dataset_name}/docs.tsv'
        (4) JSONL file containing the mapping from document ID to document title, named '{DATASET_PATH}/{dataset_name}/id2title.jsonl'.
    """

    sys.path.append(os.path.join(os.path.dirname(__file__), 'ColBERT-main'))
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Indexer
    from colbert.data import Queries
    from colbert import Searcher


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
    prediction_temp_dir = ""

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
        ranking = searcher.search_all(queries, k=k)

        title_dict = get_title_dict(doc_dataset)

        for idx, r in enumerate(ranking.data):
            ret = []
            for i, (c_id, rank, score) in enumerate(ranking.data[r]):
                doc_id = searcher.collection.pid_list[c_id]
                doc_text = searcher.collection.data[c_id]
                res_dict = get_doc(doc_dataset, doc_id, doc_text, score, title_dict, logger)
                ret.append(res_dict)
            query_data[idx]["docs"] = ret  # double check I'm putting this in the right spot

        logger.info("Search complete")

    save_file(output_file_name, query_data, logger)
    
    return

def dense_random_retrieval(
    query_data: dict,
    doc_dataset: str,
    embed_file: str,
    embed_model_name: str,
    embed_model_type: str,
    index_fn,
    dist_type,
    num_threads: int,
    text_key: str,
    index_kwargs: dict,
    output_file_name,
    logger,
    doc_dtype: pysvs.DataType = pysvs.float32,
    load_search_results=None
):
    
    """
    Dense SVS retrieval assuming the corpus of vector embeddings have already been created. Retrieves all neighbors for each query in batches. Then, randomly selects 10
    neighbors from each percentile to output. 

    It assumes these files exist:
        (1) .fvecs format vector embedding file, named '{VEC_PATH}/{embed_file}.fvecs'.
        (2) JSONL file containing the corpus of documents, named '{DATASET_PATH}/{dataset_name}/docs.jsonl'.
            - Should contain an "id" field, which is assumed to be formatted as '{document_id}_{passage_id}'
            - Should also contain a field with the passage text. Field name is specified by the "text_key" argument.
        (3) JSONL file containing the mapping from document ID to document title, named '{DATASET_PATH}/{dataset_name}/id2title.jsonl'.
    """
    queries = [d["question"] for d in query_data]  # change to question
    title_dict = get_title_dict(doc_dataset)

    k = int(len(title_dict))

    batch_size = 100
    query_batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
    query_data_batches = [query_data[i:i+batch_size] for i in range(0, len(query_data), batch_size)]


    import gc

    if not load_search_results:
        index_path = os.path.join(INDEX_PATH, 'dense', embed_file.split(".fvecs")[0])
        vec_file = os.path.join(VEC_PATH, embed_file)
        # Optimal configuration is to set the number of threads to the batch size
        index_kwargs.update({'num_threads': num_threads})

        logger.info('Start indexing...')
        search_index_og = index.dense_build_index(
            index_path,
            vec_file,
            index_fn,
            dist_type,
            doc_dtype,
            index_kwargs,
            logger=logger
        )
        logger.info('Done indexing')

        if embed_model_type == 'st':
            import sentence_transformers as st
            embed_model = st.SentenceTransformer(embed_model_name)
        else:
            raise NotImplementedError('Need to implement alternate type of embedding model')

    logger.info('Embedding and batching queries...')
    
    import datasets
    datasets.config.IN_MEMORY_MAX_SIZE = 600 * 1e9  # allow large datasets to be entirely held in memory
    # Load in the corpus to get the text associated with the retrieved documents
    corpus_file = os.path.join(DATASET_PATH, doc_dataset, "docs.jsonl")

    
    logger.info(f"Number of batches: {len(query_batches)}")
    for batch_id, queries in enumerate(tqdm(query_batches)):
        logger.info(f"Batch size: {len(queries)}")
        query_data = query_data_batches[batch_id]

        if load_search_results:
            batch_load_results = load_search_results.replace("*", str(batch_id))
            k_neighbors, dist_neighbors = load_pickle(batch_load_results, logger)
        else: 
            query_embs = embed_model.encode(queries)

            search_index = search_index_og
            logger.info(f"Start searching for {k} neighbors per query...")
            k_neighbors, dist_neighbors = search_index.search(query_embs, k)
            logger.info('Done searching')
        
            # Save direct search outputs before writing to JSON
            save_pickle([k_neighbors, dist_neighbors], f'{doc_dataset}_tmp_batch-{batch_id}.pkl', logger)
            gc.collect()

        logger.info('Loading text corpus and document titles to associate with neighbors')
        corpus = datasets.load_dataset("json", data_files=corpus_file)
        corpus = corpus['train']
        save_noise(query_data, queries, k, k_neighbors, corpus, dist_neighbors, doc_dataset, text_key, title_dict, output_file_name, logger)
    
    if not load_search_results:
        del search_index

    logger.info('DONE')
    




def dense_retrieval(
    query_data: dict,
    k: int,
    doc_dataset: str,
    embed_file: str,
    embed_model_name: str,
    embed_model_type: str,
    index_fn,
    dist_type: pysvs.DistanceType,
    num_threads: int,
    text_key: str,
    index_kwargs: dict,
    calib_kwargs: dict,
    search_win_size: int,
    output_file_name: str,
    logger: logging.Logger,
    doc_dtype: pysvs.DataType = pysvs.float32,
    load_search_results=None,
    load_index=None
):
    """
    Dense SVS retrieval assuming the corpus of vector embeddings have already been created.
    It assumes these files exist:
        (1) .fvecs format vector embedding file, named '{VEC_PATH}/{embed_file}.fvecs'.
        (2) JSONL file containing the corpus of documents, named '{DATASET_PATH}/{dataset_name}/docs.jsonl'.
            - Should contain an "id" field, which is assumed to be formatted as '{document_id}_{passage_id}'
            - Should also contain a field with the passage text. Field name is specified by the "text_key" argument.
        (3) JSONL file containing the mapping from document ID to document title, named '{DATASET_PATH}/{dataset_name}/id2title.jsonl'.
    """
    queries = [d["question"] for d in query_data]  # change to question
    title_dict = get_title_dict(doc_dataset)

    if k == 'all':  # retrieve all documents for each query
        k = len(title_dict)
        logger.info(f"All detected: {k} documents loaded")
    k = int(k)

    if load_search_results is None:
        index_path = os.path.join(INDEX_PATH, 'dense', embed_file.split(".fvecs")[0])
        if load_index:
            # A previously saved index will have saved data with it as well
            vec_file = os.path.join(index_path, 'data')
            index_kwargs.update({'config_path': index_path})
        else:
            vec_file = os.path.join(VEC_PATH, embed_file)
        # Optimal configuration is to set the number of threads to the batch size
        index_kwargs.update({'num_threads': num_threads})
        # If the following are read from the config file, convert to the correct class by eval(str)
        index_fn = eval(index_fn) if isinstance(index_fn, str) else index_fn
        dist_type = eval(dist_type) if isinstance(dist_type, str) else dist_type
        doc_dtype = eval(doc_dtype) if isinstance(doc_dtype, str) else doc_dtype

        logger.info('Start indexing...')
        search_index = index.dense_build_index(
            index_path,
            vec_file,
            index_fn,
            dist_type,
            doc_dtype,
            index_kwargs,
            calib_kwargs,
            search_win_size,
            logger
        )
        logger.info('Done indexing')
    
        logger.info('Embedding queries...')
        if embed_model_type == 'st':
            import sentence_transformers as st
            embed_model = st.SentenceTransformer(embed_model_name)
            query_embs = embed_model.encode(queries)
        else:
            raise NotImplementedError('Need to implement alternate type of embedding model')
    
        logger.info(f"Start searching for {k} neighbors per query...")
        k_neighbors, dist_neighbors = search_index.search(query_embs, k)
        logger.info('Done searching')
     
        #if calib_kwargs:
        #    # DEBUG: check how this compares to ground truth 
        #    calib_path = os.path.join(DATA_PATH, 'vamana_calib_data')
        #    calib_str = 'asqa_bge-base-dense'
        #    groundtruth = np.load(f'{calib_path}/{calib_str}_ground_truth.npy')
        #    qi = np.load(f'{calib_path}/{calib_str}_query_inds.npy')
        #    recall = pysvs.k_recall_at(groundtruth, k_neighbors[qi, :], 10, 10)
        #    logger.info(f"Sanity check: search recall is {recall} when compared to ground truth of nearest neighbors")
    
        # Save direct search outputs before writing to JSON
        save_pickle([k_neighbors, dist_neighbors], f'{doc_dataset}_tmp.pkl', logger)
        del search_index
        gc.collect()
    else:
        k_neighbors, dist_neighbors = load_pickle(load_search_results, logger)
        logger.info(f"Loaded pickle of length: {len(k_neighbors)}")

    import datasets
    datasets.config.IN_MEMORY_MAX_SIZE = 600 * 1e9  # allow large datasets to be entirely held in memory
    logger.info('Loading text corpus and document titles to associate with neighbors')
    # Load in the corpus to get the text associated with the retrieved documents
    corpus_file = os.path.join(DATASET_PATH, doc_dataset, "docs.jsonl")
    corpus = datasets.load_dataset("json", data_files=corpus_file)
    corpus = corpus['train']

    logger.info('Saving text and titles for each neighbor')
    for qi, q in enumerate(tqdm(queries)):
        neighbor_inds = k_neighbors[qi, :]
        neighbor_data = corpus[neighbor_inds]
        # Get associated text
        n_text = neighbor_data[text_key]
        # Get document & passage ID. Also get associated document title.
        n_id = neighbor_data["id"]
        ret = []
        # different output for different doc datasets
        for j, (doc_id, doc_text) in enumerate(zip(n_id, n_text)):
            score = str(dist_neighbors[qi, j])
            res_dict = get_doc(doc_dataset, doc_id, doc_text, score, title_dict, logger)
            ret.append(res_dict)
        query_data[qi]['docs'] = ret
 
    save_file(output_file_name, query_data, logger)
    logger.info('DONE')
    return


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # check that index file path env variable has been set 
    if not os.path.exists(INDEX_PATH):
        logger.info("Please set environment variable $INDEX_PATH to specify where to save or load indices.")
        raise IncompleteSetup

    logger.info(args)

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
        colbert_retrieval(
            query_data,
            args.k,
            args.query_dataset,
            args.doc_dataset,
            args.output_file,
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
        if args.noise_experiment:
            dense_random_retrieval(
                query_data,
                args.doc_dataset,
                args.embed_file,
                args.embed_model_name,
                args.embed_model_type,
                args.index_fn,
                args.dist_type,
                args.num_threads,
                args.text_key,
                args.index_kwargs,
                args.output_file,
                logger,
                args.doc_dtype,
                args.load_search_results
            )
        else: 
            dense_retrieval(
                query_data,
                args.k,
                args.doc_dataset,
                args.embed_file,
                args.embed_model_name,
                args.embed_model_type,
                args.index_fn,
                args.dist_type,
                args.num_threads,
                args.text_key,
                args.index_kwargs,
                args.calib_kwargs,
                args.search_win_size,
                args.output_file,
                logger,
                args.doc_dtype,
                args.load_search_results,
                args.load_index
            )
    else:
        print(f"Invalid retriever: {args.retriever}")
        print("Current implemented options include: colbert/dense")
        raise InvalidArgument


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--config", type=str, default=None, help="name of config file in retriever/configs/")
    parser.add_argument("--k", default=100, help="Number of nearest neighbors to retrieve, or 'all'")
    parser.add_argument("--retriever", type=str, help="options: colbert/dense")
    parser.add_argument("--query_file", type=str, help="path to the data file with queries")
    parser.add_argument("--query_dataset", type=str, help="query dataset name -> options: nq")
    parser.add_argument("--doc_dataset", type=str, help="dataset name -> options: kilt_wikipedia/// ")
    parser.add_argument("--output_file", type=str, default=None, help="Same format as the data file but with the retrieved docs; if not specified, will be added to the original data_file")
 
    # SVS arguments
    parser.add_argument("--embed_file", default=None, help='path to .fvecs vector embedding file from $VEC_PATH')
    parser.add_argument("--embed_model_name", default=None, help='model to use for embedding queries with SentenceTransformer (eg. "snowflake/snowflake-arctic-embed-s")')
    parser.add_argument("--embed_model_type", default="st", help="Type of embedding model to use, choose from [st, hf]. st is SentenceTransformers and hf is HuggingFace")
    parser.add_argument("--index_fn", default=pysvs.Flat, help='type of SVS index used for dense vector retrieval')
    parser.add_argument("--dist_type", default=pysvs.DistanceType.MIP, help='type of distance to use for index/search')
    parser.add_argument("--doc_dtype", default=pysvs.float32,
                        help='data type for the corpus vectors')
    parser.add_argument("--num_threads", type=int, default=32, help='number of threads for SVS indexing')
    parser.add_argument("--text_key", type=str, default="text", help='key in corpus data dictionary')
    parser.add_argument("--index_kwargs", type=json.loads, default="{}",
                        help='arguments for building the search index')
    parser.add_argument("--calib_kwargs", type=json.loads, default="{}",
                        help='arguments to calibrate the search index parameters')
    parser.add_argument("--search_win_size", type=int, default=None, help='size of the graph search window')
    parser.add_argument("--load_index", action="store_true", help="Load a previously built index")
    parser.add_argument("--load_search_results", type=str, default=None, help="path to pickle file with neighbors and similarity scores. Only set this argument if you have already finished search and want to skip it.")
    
    parser.add_argument("--noise_experiment", action="store_true", help="Whether to conduct retrieval for random noise experiments")
    
    args = parser.parse_args()

    # get defaults from config file
    if args.config:
        config = os.path.join(os.getcwd(), 'retriever', 'configs', args.config)
        config = yaml.safe_load(open(config))
        parser.set_defaults(**config)

    args = parser.parse_args()

    if args.noise_experiment:
        args.k = 'all'
    
    if not args.output_file:
        args.output_file = args.query_file

    main(args)
