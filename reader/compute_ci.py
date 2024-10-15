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
    parser.add_argument("-d", "--datasets", nargs='+', default=['asqa', 'nq']) 
    parser.add_argument("-f", "--subfolder", type=str, default=None) 
    parser.add_argument("-m", "--models", nargs='+', default=['Llama', 'Mistral'])
    parser.add_argument("-r", "--retrievers", nargs='+', default=['gold', 'closedbook', 'bge-base', 'colbert'])
    args = parser.parse_args()

    # Set keywords to look for in the file names
    datasets = args.datasets  #['asqa', 'nq', 'qampari']
    models = args.models  #['Llama', 'Mistral']
    conditions = args.retrievers  #['gold', 'closedbook', 'bge-base', 'colbert']
    
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
                    if os.path.exists(fname_agg):
                        print(f"{fname_agg} exists, continuing...")
                        continue
                    else:
                        stat_list = {} #= load_json(fname_agg)
                    print(fname)
                    tmp_df = load_json(fstr)
                    #tmp_df = pd.read_json(fname, encoding_errors='ignore')
                    if (dataset == 'nq') or (dataset == 'nqcite'):
                       acc = tmp_df['ragged_substring_match'] * 100
                    elif dataset == 'asqa':
                       acc = tmp_df['str_em']
                    elif dataset == 'qampari':
                       acc = tmp_df['qampari_rec_top5']
                    stat_list.update(bootstrap_ci(acc, key_prefix='em_rec'))
                    for cm in citation_measures:
                        stat_list.update(bootstrap_ci(tmp_df[cm], key_prefix=cm))
                    save_json(stat_list, fname_agg)
                    print(f"Created score file {fname_agg} with CIs using perquery data")
                    del tmp_df, acc, fname_agg
 
