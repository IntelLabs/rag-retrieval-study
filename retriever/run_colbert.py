import argparse
import sys
import logging
import yaml

# import indexing functions for all retrievers
from retriever.ret_utils import *
from utils import InvalidArgument, IncompleteSetup
from file_utils import load_json

DATA_PATH = os.environ.get("DATA_PATH")
DATASET_PATH = os.environ.get("DATASET_PATH")
COLBERT_MODEL_PATH = os.environ.get("COLBERT_MODEL_PATH")
INDEX_PATH = os.environ.get("INDEX_PATH")
VEC_PATH = os.environ.get("VEC_PATH")


def colbert_build_index(
        config,
        model_dir,
        doc_dataset,
        doc_dataset_path
):
    """
    Builds colbert index with colbertv2.0 for doc dataset
    """
    from colbert import Indexer

    indexer = Indexer(
        checkpoint=model_dir,
        config=config
    )
    indexer.index(
        name=doc_dataset,
        collection=doc_dataset_path,
        overwrite='reuse'
    )


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
        colbert_build_index(
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

        try:
            for idx, r in enumerate(ranking.data):
                ret = []
                for i, (c_id, rank, score) in enumerate(ranking.data[r]):
                    doc_id = searcher.collection.pid_list[c_id]
                    doc_text = searcher.collection.data[c_id]
                    res_dict = get_doc(doc_dataset, doc_id, doc_text, score, title_dict, logger)
                    ret.append(res_dict)
                query_data[idx]["docs"] = ret  # double check I'm putting this in the right spot
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

        logger.info("Search complete")

    save_file(output_file_name, query_data, logger)
    
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


    colbert_retrieval(
        query_data,
        args.k,
        args.query_dataset,
        args.doc_dataset,
        args.output_file,
        logger
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--config", type=str, default=None, help="name of config file in retriever/configs/")
    parser.add_argument("--k", default=100, help="Number of nearest neighbors to retrieve, or 'all'")
    parser.add_argument("--retriever", type=str, help="options: colbert/dense")
    parser.add_argument("--query_file", type=str, help="path to the data file with queries")
    parser.add_argument("--query_dataset", type=str, help="query dataset name -> options: nq")
    parser.add_argument("--doc_dataset", type=str, help="dataset name -> options: kilt_wikipedia/// ")
    parser.add_argument("--output_file", type=str, default=None, help="Same format as the data file but with the retrieved docs; if not specified, will be added to the original data_file")
    
    args = parser.parse_args()

    # get defaults from config file
    if args.config:
        config = os.path.join(os.getcwd(), 'retriever', 'configs', args.config)
        config = yaml.safe_load(open(config))
        parser.set_defaults(**config)

    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = args.query_file

    main(args)
