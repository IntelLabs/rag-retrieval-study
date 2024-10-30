import argparse
import datasets
import json
import numpy as np
import os
import time
import tqdm

# Import from the current codebase
from file_utils import load_jsonl, save_json

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(input_file, output_prefix, corpus_path, title_json, text_key):
    # Converts the NQ dataset, as obtained from the RAGGED repository (https://github.com/neulab/ragged), into the format needed for dense vector retrieval and evaluation of the retrieval results.

    query_data = load_jsonl(input_file, sort_by_id=False)

    # Query file needs fields: question, answer, id/sample_id, other metadata
    # Gold file needs: all query fields, plus 'docs' field
    def extract_provenance(p, title, page, par):
        if (p['start_par_id'] == 'None') or (p['end_par_id'] == 'None'):
            # Sometimes this happens in the bioasq dataset when evidence is in a table or figure
            logger.warning('Evidence is not within text -- dropping!')
            return title, page, par
        elif p['start_par_id'] == p['end_par_id']:
            par.append(f'{p['page_id']}_{p['start_par_id']}')
        else:
            logger.warning('Gold answer spans multiple paragraphs')
            for pii in range(int(p['start_par_id']), int(p['end_par_id']) + 1):
                par.append(f'{p["page_id"]}_{pii}')
        title.append(p['title'])
        page.append(p['page_id'])
        return title, page, par

    out_queries = []
    for i, query in tqdm.tqdm(enumerate(query_data)):
        q_dict = {'id': query['id'], 'question': query['input']}
        ans, title, page, par = [], [], [], []
        for j, qo in enumerate(query['output']):
            if qo['answer']:
                ans.append(qo['answer'])
            if isinstance(qo['provenance'], list):
                for p in qo['provenance']:
                    title, page, par = extract_provenance(p, title, page, par)
            elif isinstance(qo['provenance'], dict):
                title, page, par = extract_provenance(qo['provenance'], title, page, par)
        q_dict.update({'output': {'answer_set': ans,
                                  'title_set': list(set(title)),
                                  'page_id_set': list(set(page)),
                                  'page_par_id_set': list(set(par))}
                       })
        q_dict.update({'answer': ans[0]})  # just a simplifying shortcut to grab one answer for calculating rouge
        out_queries.append(q_dict)
    del ans, title, page, par
    filename = f'{output_prefix}_qa.json'
    save_json(out_queries, filename)
    logger.info(f"Saved queries to {filename}")

    # Load in the corpus to get the text associated with the gold documents
    t0 = time.time()
    if 'json' in corpus_path:
        corpus = datasets.load_dataset("json", data_files=corpus_path)
        corpus = corpus['train']
    else:
        # This should already exist if vector embeddings have been generated
        corpus = datasets.load_from_disk(corpus_path)
    id2title = datasets.load_dataset("json", data_files=title_json)
    id2title = id2title['train']
    logger.info(f"Loading corpus and document IDs took {time.time() - t0} seconds")

    # To save on search/indexing time, filter the datasets to only grab the rows we want
    all_pars = []
    all_pages = []
    for out_dict in out_queries:
        all_pars += out_dict['output']['page_par_id_set']
        all_pages += out_dict['output']['page_id_set']
    all_pars = np.array(all_pars)
    all_pages = np.array(all_pages)
    use_int = False
    if isinstance(id2title[0]['id'], int):
        use_int = True
        all_pages = all_pages.astype(int)
    t0 = time.time()
    text_rows = corpus.filter(lambda x: np.isin(x['id'], all_pars), batched=True, batch_size=30720)
    id_rows = id2title.filter(lambda x: np.isin(x['id'], all_pages), batched=True, batch_size=30720)
    all_ids = np.array(text_rows['id'])
    title_ids = np.array(id_rows['id'])
    del corpus, id2title
    logger.info(f"Filtering corpus and document IDs took {time.time() - t0} seconds")

    # Now actually get the associated gold document title and passage text and update the dict
    skipped = 0
    for i, out_dict in enumerate(tqdm.tqdm(out_queries)):
        docs = []
        for par in out_dict['output']['page_par_id_set']:
            page, para = par.split('_')
            page = int(page) if use_int else page
            id_ind = np.where(title_ids == page)[0]
            text_ind = np.where(all_ids == par)[0]
            if len(text_ind) == 0:
                # This happens sometimes in bioasq when the evidence is within a figure or table, not text
                logger.warning(f"Could not find paragraph matching {par}, skipping!")
                skipped += 1
                continue
            thisdoc = {'id': par,
                       'title': id_rows[id_ind]['title'][0],
                       'text': text_rows[text_ind][text_key][0]}
            docs.append(thisdoc)
        out_dict.update({'docs': docs})
    logger.warning(f"Skipped a total of {skipped} passages that had IDs but no associated text")

    filename = f'{output_prefix}_gold.json'
    save_json(out_queries, filename)
    print(f"Saved dataset and documents to {filename}")


if __name__ == '__main__':
    # Example usage:
    # python convert_ragged-json_to_alce-json.py --input_file /export/data/vyvo/rag/datasets/nq.jsonl \
    #   --output_prefix nq \
    #   --corpus_path /export/data/vyvo/hf_cache_overflow/processed_kilt_wikipedia \
    #   --title_json /export/data/vyvo/rag/datasets/kilt_wikipedia/kilt_wikipedia_jsonl/id2title.jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--title_json', type=str)

    args = vars(parser.parse_args())
    DATA_PATH = os.environ.get("DATA_PATH")
    args['output_prefix'] = f'{DATA_PATH}/{args["output_prefix"]}'

    print(args)

    # Allow large datasets to be entirely held in memory
    datasets.config.IN_MEMORY_MAX_SIZE = 600 * 1e9

    if 'kilt' in args['corpus_path']:
        text_key = 'paras'

    main(**args, text_key=text_key)
