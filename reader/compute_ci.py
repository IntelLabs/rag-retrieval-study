# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import pathlib
import os
from file_utils import load_json, save_json


RESULTS_PATH = os.environ.get("RESULTS_PATH")

def bootstrap_ci(data, key_prefix, func=np.mean, n_boot=1000, percentiles=[2.5, 97.5]):
    stats = np.empty((n_boot,))
    for i in range(n_boot):
        sample = np.random.choice(data, size=len(data))
        stats[i] = func(sample)
    overall = func(stats)
    ci = np.percentile(stats, percentiles)
    return {f'{key_prefix}_mean': overall, f'{key_prefix}_ci_lower': ci[0], f'{key_prefix}_ci_upper': ci[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", nargs='+', default=['asqa', 'nq'],
                        help="Dataset(s) in the score filename") 
    parser.add_argument("-f", "--subfolder", type=str, default=None,
                        help="Searches in a subfolder in the RESULTS_PATH") 
    parser.add_argument("-m", "--models", nargs='+', default=['Llama', 'Mistral'],
                        help="Model(s) in the score filename")
    parser.add_argument("-c", "--conditions", nargs='+', default=['gold', 'closedbook', 'bge-base', 'colbert'],
                        help="Condition(s) in the score filename")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Set keywords to look for in the file names
    datasets = args.datasets  #['asqa', 'nq', 'qampari']
    models = args.models  #['Llama', 'Mistral']
    conditions = args.conditions
    
    # Figure out which citation metrics to compute CIs for, in addition to Exact Match Recall
    citation_measures = ['citation_rec', 'citation_prec']
    
    for dataset in datasets:
        for model in models:
            for cond in conditions:
                if args.subfolder:
                    score_list = pathlib.Path(RESULTS_PATH + f'/reader/{args.subfolder}').rglob(f'{dataset}*{model}*{cond}*perquery.score')
                else:
                    score_list = pathlib.Path(RESULTS_PATH + '/reader').rglob(f'{dataset}*{model}*{cond}*perquery.score')
                for fname in score_list:
                    fstr = str(fname)
                    fname_agg = fstr[:-20] + '-mean-ci.score'
                    if os.path.exists(fname_agg) and (not args.overwrite):
                        print(f"{fname_agg} exists, continuing...")
                        continue
                    else:
                        stat_list = {}
                    print(fname)
                    tmp_df = load_json(fstr)
                    #tmp_df = pd.read_json(fname, encoding_errors='ignore')
                    if (dataset == 'nq') or (dataset == 'nqcite'):
                        acc = np.array(tmp_df['ragged_substring_match']) * 100
                    elif dataset == 'asqa':
                        acc = np.array(tmp_df['str_em'])
                    elif dataset == 'qampari':
                        acc = np.array(tmp_df['qampari_rec_top5'])
                    stat_list.update(bootstrap_ci(acc, key_prefix='em_rec'))
                    if 'shot0' not in fstr:  # Skip computing citation metrics if this is the vanilla NQ task
                        for cm in citation_measures:
                            stat_list.update(bootstrap_ci(tmp_df[cm], key_prefix=cm))
                    save_json(stat_list, fname_agg)
                    print(f"Created score file {fname_agg} with CIs using perquery data")
                    del fstr, tmp_df, acc, fname_agg
 
