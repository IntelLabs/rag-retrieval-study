# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Some of this code is based on prior work under the MIT License:
#   Copyright (c) 2023 Princeton Natural Language Processing

import json
import os 
import warnings
import numpy as np
import pdb
import sys
import argparse
import matplotlib.pyplot as plt
import logging

DATA_PATH = os.environ.get("DATA_PATH")
RESULTS_PATH = os.environ.get("RESULTS_PATH")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from file_utils import save_json, load_json
from reader.compute_ci import bootstrap_ci


def count_jsonl(filename):
    with open(filename, 'r') as f:
        count = sum(1 for _ in f)
    return count


def get_precision(guess_page_id_set, gold_page_id_set):
    precision = np.mean([[s in gold_page_id_set] for s in guess_page_id_set])
    return precision


def get_recall(guess_page_id_set, gold_page_id_set):
    recall = np.mean([[s in guess_page_id_set] for s in gold_page_id_set]) if len(gold_page_id_set) > 0 else 0.0
    return recall


def get_retriever_results(eval_data, par_level, plot=True):
        
    retriever_results = []
    for sample_id, (d) in enumerate(eval_data):
        output = d['output']
        if par_level:
            ### starts being different 
            id = d['id']
            gold_page_ids = output[f'page_id_set']
            gold_page_par_ids = output[f'page_par_id_set']
            gold_answer_set = output['answer_set']
            
            # For each retrieved document, get wiki and par id match
            doc_retriever_results = []

            for p in d['docs']:
                guess_page_id = p['id']
                doc_retriever_result = {}
                doc_retriever_result['page_id'] = guess_page_id
                doc_retriever_result['page_id_match'] = guess_page_id in gold_page_ids

                guess_page_par_id = guess_page_id + '_' + str(p['start_par_id'])
                doc_retriever_result['answer_in_context'] = any([ans in p['text'] for ans in gold_answer_set])
                doc_retriever_result[f'page_par_id'] = guess_page_par_id
                doc_retriever_result[f'page_par_id_match'] = guess_page_par_id in gold_page_par_ids
                doc_retriever_results.append(doc_retriever_result)

            retriever_result = {}
            retriever_result['id'] = id
            retriever_result['gold provenance metadata'] = {}
            retriever_result['gold provenance metadata']['num_page_ids'] = len(gold_page_ids)
            retriever_result['gold provenance metadata']['num_page_par_ids'] = len(gold_page_par_ids)
            retriever_result['passage-level results'] = doc_retriever_results
            retriever_results.append(retriever_result)

        else:
            gold_ids = output['id_set']
            doc_retriever_results = []

            for p in d['docs']:
                guess_id = str(p['id'])
                doc_retriever_result = {}
                doc_retriever_result['id'] = guess_id
                doc_retriever_result['id_match'] = guess_id in gold_ids
                doc_retriever_results.append(doc_retriever_result)

            retriever_result = {}
            retriever_result['id'] = sample_id
            retriever_result['gold provenance metadata'] = {}
            retriever_result['gold provenance metadata']['num_ids'] = len(gold_ids)
            retriever_result['passage-level results'] = doc_retriever_results
            retriever_results.append(retriever_result)

    return retriever_results


def print_retriever_acc(retriever_results, gold_data, par_level, ci=False):
    if par_level:
        page_id_per_r = []
        page_par_id_per_r = []
        answer_in_context_per_r = []

        for r in retriever_results:
            page_ids = []
            page_par_ids = []
            answer_in_context = []
            
            for d in r['passage-level results']:
                page_ids.append(d[f'page_id'])
                page_par_ids.append(d[f'page_par_id'])
                answer_in_context.append(d['answer_in_context'])

            page_id_per_r.append(page_ids)
            page_par_id_per_r.append(page_par_ids)
            answer_in_context_per_r.append(answer_in_context)

        ks = np.arange(1, len(page_id_per_r[0])+1)
        results_by_k, results_lists = {}, {}
        for k in ks:
            results_by_k[int(k)] = {}
            results_lists[(int)(k)] = {
                f'top-k accuracy page_id': [],\
                f'top-k accuracy page_par_id': [],\
                f"precision@k page_id": [],\
                f"precision@k page_par_id": [],\
                f"recall@k page_id": [],\
                f"recall@k page_par_id": [],\
                'answer_in_context@k': []
            }
            for r in range(len(retriever_results)):

                gold_page_id_set = gold_data[r]['output'][f'page_id_set']
                gold_page_par_id_set = gold_data[r]['output'][f'page_par_id_set']
                guess_page_id_set = set(page_id_per_r[r][:k])
                guess_page_par_id_set = set(page_par_id_per_r[r][:k])
                
                results_lists[(int)(k)][f'top-k accuracy page_id'].append(any([(w in gold_page_id_set) for w in guess_page_id_set]))
                results_lists[(int)(k)][f'top-k accuracy page_par_id'].append(any([(w in gold_page_par_id_set) for w in guess_page_par_id_set]))
                results_lists[(int)(k)][f"precision@k page_id"].append(get_precision(guess_page_id_set, gold_page_id_set))
                results_lists[(int)(k)][f"precision@k page_par_id"].append(get_precision(guess_page_par_id_set, gold_page_par_id_set))
                results_lists[(int)(k)][f"recall@k page_id"].append(get_recall(guess_page_id_set, gold_page_id_set))
                results_lists[(int)(k)][f"recall@k page_par_id"].append(get_recall(guess_page_par_id_set, gold_page_par_id_set))
                results_lists[(int)(k)]["answer_in_context@k"].append(any(answer_in_context_per_r[r][:k]))

            for key, val in results_lists[(int)(k)].items():
                arrval = np.array(val)
                if ci:
                    ci_dict = bootstrap_ci(arrval, key)
                    results_by_k[int(k)].update(ci_dict)
                else:
                    results_by_k[(int)(k)][key] = arrval.mean()
    else:
        id_per_r = []

        for r in retriever_results:
            ids = []
            for d in r['passage-level results']:
                ids.append(d[f'id'])
            id_per_r.append(ids)

        ks = np.arange(1, len(id_per_r[0])+1)
        results_by_k, results_lists = {}, {}
        for k in ks:
            results_by_k[int(k)] = {}
            results_lists[(int)(k)] = {
                f'top-k accuracy id': [],\
                f"precision@k id": [],\
                f"recall@k id": []
            }
            for r in range(len(retriever_results)):

                gold_id_set = gold_data[r]['output'][f'id_set']
                guess_id_set = set(id_per_r[r][:k])
                
                results_lists[(int)(k)][f'top-k accuracy id'].append(any([(w in gold_id_set) for w in guess_id_set]))
                results_lists[(int)(k)][f"precision@k id"].append(get_precision(guess_id_set, gold_id_set))
                results_lists[(int)(k)][f"recall@k id"].append(get_recall(guess_id_set, gold_id_set))

            for key, val in results_lists[(int)(k)].items():
                arrval = np.array(val)
                if ci:
                    ci_dict = bootstrap_ci(arrval, key)
                    results_by_k[int(k)].update(ci_dict)
                else: 
                    results_by_k[(int)(k)][key] = arrval.mean()

    return results_by_k, ks


def results_by_key(ks, results_by_k, metric, par_level=True):
    metric = metric.strip()
    if par_level:
        page_id_match = []
        page_par_id_match = []
        for k in ks:
            page_id_match.append(results_by_k[k][f"{metric} page_id"])
            page_par_id_match.append(results_by_k[k][f"{metric} page_par_id"])
        return page_id_match, page_par_id_match
    else:
        id_match = []
        for k in ks:
            id_match.append(results_by_k[k][f"{metric} id"])
        return id_match


def generate_plot(ks, results_by_k, plt_title, out_file, metric, par_level):
    plt.cla()
    plt.title(plt_title)
    plt.ylabel(metric)

    scatter_ks = [1, 2, 3, 4, 5, 10, 20, 100]

    if par_level: 
        page_id_match, page_par_id_match = results_by_key(ks, results_by_k, metric)
        plt.plot(ks, page_id_match, 'b--', label = 'page_id')
        plt.plot(ks, page_par_id_match, 'b', label = 'page_par_id')
        plt.legend()

        page_id_scatter = [v for k, v in zip(ks, page_id_match) if k in scatter_ks]
        page_par_id_scatter = [v for k, v in zip(ks, page_par_id_match) if k in scatter_ks]

        plt.scatter(scatter_ks, page_id_scatter)
        plt.scatter(scatter_ks, page_par_id_scatter)


    else:
        id_match = results_by_key(ks, results_by_k, metric, par_level)
        plt.plot(ks, id_match)

        id_scatter = [v for k, v in zip(ks, id_match) if k in scatter_ks]
        plt.scatter(scatter_ks, id_scatter)

    plt.xlabel('k')
    plt.savefig(out_file)
    plt.show()


def plot_sim_scores(eval_data, plt_title, out_file, par_level):
    """
    Plots average similarity score between query and document by neighbor id
    """

    if par_level:
        gold_doc_sim = []  # track similarity scores of all gold documents and paragraphs
        gold_par_sim = []
    else: 
        gold_sim = []  # track similarity scores of all gold documents

    neighbor_sim = [[] for i in range(100)]  # 100 empty lists to track neighbor scores

    for entry in eval_data:
        if par_level: 
            gold_doc_ids = entry['output']['page_id_set']
            gold_par_ids = entry['output']['page_par_id_set']
        else: 
            gold_ids = entry['output']['id_set']

        for i, d in enumerate(entry['docs']):
            doc_id = str(d['id'])
            doc_sim = float(d['score'])

            # include similarity score if document in gold data
            if par_level:
                par_id = doc_id + "_" + str(d['start_par_id'])
                if doc_id in gold_doc_ids:
                    gold_doc_sim.append(doc_sim)

                if par_id in gold_par_ids:
                    gold_par_sim.append (doc_sim)
            else: 
                if doc_id in gold_ids:
                    gold_sim.append(doc_sim)

            neighbor_sim[i].append(doc_sim)

    # get average by neighbor index 
    neighbor_avs = [sum(n_list)/len(n_list) for n_list in neighbor_sim]
    plt.cla()
    plt.title(plt_title)
    plt.ylabel("Average Similarity Score")
    plt.xlabel("Neighbor Index")

    n_idx = range(1, 101)
    plt.plot(n_idx, neighbor_avs)

    scatter_ks = [1, 2, 3, 4, 5, 10, 20, 100]
    id_scatter = [av for idx, av in enumerate(neighbor_avs) if (idx+1) in scatter_ks]
    plt.scatter(scatter_ks, id_scatter)

    if par_level:
        # line for par-level, line for doc-level
        gold_doc_av = sum(gold_doc_sim) / len(gold_doc_sim)
        gold_par_av = sum(gold_par_sim) / len(gold_par_sim)
        gold_doc_line = [gold_doc_av] * 100
        gold_par_line = [gold_par_av] * 100
        plt.plot(n_idx, gold_doc_line, label="score of gold documents")
        plt.plot(n_idx, gold_par_line, label="score of gold paragraphs")
    else: 
        gold_av = sum(gold_sim) / len(gold_sim)
        gold_line = [gold_av] * 100
        plt.plot(n_idx, gold_line, label="score of gold passages")

    plt.legend()
    plt.savefig(out_file)
    plt.show()

def main(args):
    
    eval_path = os.path.join(DATA_PATH, args.eval_file)
    eval_data = load_json(eval_path, logger)
    evaluation_dir = os.path.join(
        RESULTS_PATH,
        "retriever",
        args.eval_file.split(".")[0]
    )
    os.makedirs(evaluation_dir, exist_ok=True)

    par_level = True
    if args.not_par_level:
        par_level = False

    file_prefix = args.eval_file.split('.json')[0]  # make json with same name as input file
    # Plot neighbor similarity values
    if not args.no_plot:
        out_file = os.path.join(evaluation_dir, f"{file_prefix}_neighbor_sim.jpg")
        plot_sim_scores(eval_data, file_prefix, out_file, par_level)
        logger.info(f'saving figure in {out_file}')

    json_fn = os.path.join(evaluation_dir, file_prefix + "_results.json")
    if os.path.exists(json_fn):
        logger.info("Loading existing JSON file with retriever results...")
        par_retriever_results = load_json(json_fn, logger=logger)
    else:
        par_retriever_results = get_retriever_results(eval_data, par_level, plot=not args.no_plot)
        save_json(par_retriever_results, json_fn, logger)  

    results_by_k, ks = print_retriever_acc(par_retriever_results, eval_data, par_level, args.ci)
    if args.ci:
        save_json(results_by_k, os.path.join(evaluation_dir, f'{file_prefix}_results_by_k_ci.json'), logger)
    else:
        save_json(results_by_k, os.path.join(evaluation_dir, f'{file_prefix}_results_by_k.json'), logger)

    # Plot metrics with varying k 
    if not args.no_plot:
        for m in ["top-k accuracy", "precision@k", "recall@k"]:
            out_file = os.path.join(evaluation_dir, f"{file_prefix}_{m.replace(' ', '_')}_results_by_k.jpg")
            generate_plot(ks, results_by_k, file_prefix, out_file, m, par_level)
            logger.info(f'saving figure in {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--eval_file", help="Eval file name in $DATA_DIR containing gold and predictions")
    parser.add_argument("--not_par_level", action="store_true", help="Whether gold data is divided by document, paragraph")
    parser.add_argument("--no_plot", action="store_true", help="Omit the plots")
    parser.add_argument("--ci", action="store_true", help="Compute bootstrap 95% CIs")
    args = parser.parse_args()
    main(args)
