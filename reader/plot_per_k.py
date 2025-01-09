# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import logging
import matplotlib.pyplot as plt
from file_utils import load_json

RESULTS_PATH = os.environ.get("RESULTS_PATH")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
script to generate per-k reader result plots on same axis as retrieval results
"""

def main(args):

    metrics = []
    if 'asqa' in args.eval_file:
        metrics = ['mauve', 'str_em', 'citation_rec', 'citation_prec']
    elif 'qampari' in args.eval_file:
        metrics = ['qampari_rec_top5', 'qampari_prec', 'citation_rec', 'citation_prec']
    else:
        raise NotImplementedError

    
    file_path = os.path.join(RESULTS_PATH, "reader")
    ks = [1, 2, 3, 4, 5, 10, 20, 100]
    results = []

    if args.ret_file:
        from retriever.eval import get_retriever_results, print_retriever_acc, results_by_key
        DATA_PATH = os.environ.get("DATA_PATH")
        par_level = False

        ret_path = os.path.join(DATA_PATH, args.ret_file)
        ret_data = load_json(ret_path, logger)
        par_retriever_results = get_retriever_results(ret_data, par_level)
        results_by_k, ret_ks = print_retriever_acc(par_retriever_results, ret_data, par_level)
        id_match = results_by_key(ret_ks, results_by_k, args.ret_metric, par_level)
        ret_x = [item * 100 for item in id_match]
        ret_scatter = [v for k, v in zip(ret_ks, ret_x) if k in ks]


    for k in ks:
        filename = args.eval_file.replace("*", str(k))
        curr_path = os.path.join(file_path, filename)
        curr_dict = load_json(curr_path, logger)
        results.append(curr_dict)

    gold_file = args.eval_file.split("-")[:-2]
    gold_file = "-".join(gold_file) + "-cite-gold.json.score"
    gold_file = gold_file.replace("*", "5")
    gold_file = os.path.join(file_path, gold_file)
    gold_dict = load_json(gold_file, logger)
    # 'results/reader/asqa-Mistral-7B-Instruct-v0.3-None-shot2-42-closedbook.json.score
    # qampari-Mistral-7B-Instruct-v0.3-None-shot2-ndoc0-closedbook.json.score
    # qampari-Mistral-7B-Instruct-v0.3-None-shot2-ndoc0-42-closedbook.json.score
    no_context = args.eval_file.split("-")[:-3]
    no_context = "-".join(no_context) + "-closedbook.json.score"
    no_context = no_context.replace("*", "0")
    no_context = os.path.join(file_path, no_context)
    no_context_dict = load_json(no_context, logger)

    file_path = os.path.join(file_path, "plots")
    os.makedirs(file_path, exist_ok=True)
    for m in metrics:
        y_list = []
        for r in results:
            y_list.append(r[m])

        plot_title = args.eval_file.replace(".json.score", "")
        plot_title = plot_title.replace("ndoc*-", "")
        plot_title = plot_title.replace("/", "_")
        plt.cla()
        plt.title(plot_title)
        plt.xlabel("Number of Neighbors")
        plt.ylabel(m)
        plt.plot(ks, y_list, label="reader")
        plt.scatter(ks, y_list)

        if args.ret_file:
            plt.plot(ret_ks, ret_x, label=f"retriever {args.ret_metric}")
            plt.scatter(ks, ret_scatter)

        # add oracle line
        gold_val = gold_dict[m]
        plt.axhline(y = gold_val, color='g', linestyle='--',  label="gold")

        if 'citation' not in m:
            no_context_val = no_context_dict[m]
            plt.axhline(y=no_context_val,  color='r', linestyle='--', label="no context")
        
        plt.legend()
        plot_title = f"{plot_title}_{m}.png"
        plot_path = os.path.join(file_path, plot_title)
        logger.info(f"Saving plot to: {plot_path}")
        plt.savefig(plot_path)
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create plots for reader results")
    parser.add_argument("--eval_file", help="Eval file name in $RESULTS_PATH directory. Replace ndocs int with '*'")
    parser.add_argument("--ret_file", help="To evaluate retrieval on the same axis, provide an eval file with retrieval")
    parser.add_argument("--ret_metric", help="Retrieval metric to include on axis: 'top-k accuracy', 'precision@k', 'recall@k'")
    
    args = parser.parse_args()
    main(args)