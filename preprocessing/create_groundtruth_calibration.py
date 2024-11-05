# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
After exhaustive search with a flat search index has been run, you will have files that contain the nearest neighbors
for every single query.

To use this for calibrating the approximate nearest neighbors search, these JSON outputs need to be converted back into
numpy files. This script will select a random subset of the queries for the calibration.
"""
import argparse
import datasets as ds
import numpy as np
import os

import sentence_transformers as st

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from file_utils import load_json

DATA_PATH = os.environ.get("DATA_PATH")
DATASET_PATH = os.environ.get("DATASET_PATH")

def main(exact_search_results, dataset_path, k, embed_model_name):
    # Load the query data and grab what we need
    res = load_json(os.path.join(DATA_PATH, exact_search_results))
    nqueries = len(res)
    ncalib = nqueries // 3
    calib_inds = np.random.choice(np.arange(0, nqueries), ncalib, replace=False)
    #calib_inds = np.random.randint(low=0, high=nqueries, size=ncalib)
    # Set the output directory
    output_dir = os.path.join(DATA_PATH, 'vamana_calib_data')
    dataset_str, remain_str = exact_search_results.split('_')
    retriever_str = '-'.join(remain_str.split('-')[1:])[:-5]
    # Write the query indices to a file
    qi_path = os.path.join(output_dir, f'{dataset_str}_{retriever_str}_query_inds.npy')
    np.save(qi_path, calib_inds)
    # NQ: Each query has dict_keys(['id', 'question', 'output', 'answer', 'docs'])
    # ASQA: dict_keys(['qa_pairs', 'wikipages', 'annotations', 'question', 'answer', 'docs', 'output', 'id'])
    calib_queries = [res[ii]['question'] for ii in calib_inds]
    gt_info = [res[ii]['docs'] for ii in calib_inds]
    all_gt_ids = []
    all_gt_scores = []
    for gt_dict in gt_info:
        if 'start_par_id' in gt_dict[0]:
            gt_ids = [f"{d['id']}_{d['start_par_id']}" for d in gt_dict]
        else:
            gt_ids = [d['id'] for d in gt_dict]
        all_gt_scores.append([float(d['score']) for d in gt_dict])
        all_gt_ids.append(gt_ids)
    all_gt_ids = np.array(all_gt_ids)
    all_gt_scores = np.array(all_gt_scores)
    del res, gt_info

    # Now find the INDEX associated with these gt_ids
    if 'json' in dataset_path:
        corpus = ds.load_dataset('json', data_files=os.path.join(DATASET_PATH, dataset_path))
        corpus = corpus['train']
    else:
        corpus = ds.load_from_disk(dataset_path)

    def add_index(examples, idx):
        examples['index'] = idx
        return examples

    corpus = corpus.map(add_index, with_indices=True, batched=True)
    text_rows = corpus.filter(lambda x: np.isin(x['id'], all_gt_ids), batched=True, batch_size=30720)
    text_inds, corpus_ids = np.array(text_rows['index']), np.array(text_rows['id'])
    query_ground_truth = np.zeros((ncalib, k), dtype=np.uint32)
    for ci in range(ncalib):
        these_inds = np.concatenate([text_inds[np.where(corpus_ids == k_ind)[0]] for k_ind in all_gt_ids[ci, :k]])
        query_ground_truth[ci, :] = these_inds
    gt_path = os.path.join(output_dir, f'{dataset_str}_{retriever_str}_ground_truth.npy')
    np.save(gt_path, query_ground_truth)
    print(f'Saved ground truth to {gt_path}!')
    # Embed the queries
    embed_model = st.SentenceTransformer(embed_model_name)
    query_embs = embed_model.encode(calib_queries)
    q_path = os.path.join(output_dir, f'{dataset_str}_{retriever_str}_queries.npy')
    np.save(q_path, query_embs)
    print(f'Saved query embeddings to {q_path}!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exact_search_results', type=str,
                        default='nq_retrieval-bge-base-dense.json',
                        help='Path to JSON file containing exact search results')
    parser.add_argument('--corpus_path', type=str,
                        default='/export/data/vyvo/hf_cache_overflow/processed_kilt_wikipedia',
                        help='Path to corpus dataset (saved to disk with HF datasets)')
    parser.add_argument('--k', type=int, default=100,
                        help='Top k to record for ground truth')
    parser.add_argument("--embed_model_name", type=str,
                        default='BAAI/bge-base-en-v1.5',
                        help='model to use for embedding queries with SentenceTransformer or HuggingFace')
    parser.add_argument("--embed_model_type", default="st",
                        help="Type of embedding model to use, choose from [st, hf]. st is SentenceTransformers and hf is HuggingFace")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for picking subset of queries')
    args = parser.parse_args()

    if args.embed_model_type == 'hf':
        raise NotImplementedError("Haven't implemented HF embedding method yet")

    np.random.seed(args.seed)
    main(args.exact_search_results, args.corpus_path, args.k, args.embed_model_name)
