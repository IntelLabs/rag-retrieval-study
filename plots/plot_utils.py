# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import os
import pandas as pd
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from file_utils import load_json

RESULTS_PATH = os.environ.get("RESULTS_PATH")


def set_plot_params(plot_dict=None, seaborn_context="paper"):
    """
    Sets the general plotting parameters.
    Here default seaborn plotting context assumes you are generating images for a paper. THIS DIFFERS FROM SEABORN ITSELF, which assumes the default context is "notebook" (i.e. exploratory data analysis).
    """
    if plot_dict is None:
        plot_dict = {'figure.dpi': 300,
                     'lines.linewidth': 1.5,
                     'legend.fontsize': 8,
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'axes.spines.bottom': False,
                     'axes.spines.left': False}
    sns.set_theme(context=seaborn_context, style="darkgrid")
    matplotlib.rcParams.update(plot_dict)

def load_metric_ci_files(datasets, models, conditions, subfolder=None):
    """
    Utility that will find all the *mean-ci.score files for reporting means and 95% confidence intervals for some metrics. Will also return additional fields that differ between files: the dataset, language model, and any `condition` that you specify.
    """
    base_path = RESULTS_PATH + 'reader' if subfolder is None else RESULTS_PATH + f'/reader/{subfolder}/'
    field_list = []
    file_list = []
    for dataset in datasets:
        for model in models:
            for cond in conditions:
                extra_fields = {'dataset': dataset.upper(),
                                'model': model,
                                'condition': cond}
                file_iter = pathlib.Path(base_path).rglob(f'{dataset}-{model}-*{cond}-*mean-ci.score')
                for f in file_iter:
                    fstr = str(f)
                    file_list.append(fstr)
                    if 'shot0' in str(f):
                        # Handles the NQ dataset when citations are not requested
                        field_list.append({'dataset': f'{dataset.upper()}-nocite',
                                           'model': model,
                                           'condition': cond})
                    else:
                        field_list.append(extra_fields)
    return file_list, field_list

def load_metric_ndoc_files(datasets, models, retrievers, conditions):
    """
    Utility that will find all the .score files for reporting reader recall with n docs in prompt.
    """
    res_path = RESULTS_PATH + '/reader'
    field_list = []
    file_list = []
    for dataset in datasets:
        for model in models:
            for retriever in retrievers:
                for cond in conditions:
                    base_path = res_path + f'/{dataset}_ndoc/'
                    extra_fields = {
                        'dataset': dataset,
                        'model': model,
                        'retriever': retriever,
                        'condition': cond
                    }

                    if dataset == 'nq-nocite':
                        temp_dataset = 'nq'
                        file_iter = pathlib.Path(base_path).rglob(f'{temp_dataset}-{model}-*shot0-ndoc{cond}-*{retriever}*mean-ci.score')
                    else:
                        file_iter = pathlib.Path(base_path).rglob(f'{dataset}-{model}-*shot[1,2]-ndoc{cond}-*{retriever}*mean-ci.score')
                    
                    for f in file_iter:
                        fstr = str(f)
                        file_list.append(fstr)
                        field_list.append(extra_fields)

            for s_cond in ['gold', 'no-context', 'closedbook']:
                if dataset == 'nq-nocite':
                    temp_dataset = 'nq'
                    file_iter = pathlib.Path(base_path).rglob(f'{temp_dataset}-{model}-*shot0-*-{s_cond}*mean-ci.score')
                else:   
                    file_iter = pathlib.Path(base_path).rglob(f'{dataset}-{model}-*shot[1,2]*-{s_cond}*mean-ci.score')
                if s_cond == 'closedbook':
                    s_cond = 'no-context'  # rename for consistency
                extra_fields = {
                    'dataset': dataset,
                    'model': model,
                    'retriever': None,
                    'condition': s_cond
                }
                for f in file_iter:
                    fstr = str(f)
                    file_list.append(fstr)
                    field_list.append(extra_fields)

    return file_list, field_list

def load_metric_noise_files(datasets, models, retrievers, conditions, subfolder=None):
    """
    Utility that will find all the .score files for reporting reader recall with noise added to the retriever.
    """
    base_path = RESULTS_PATH + '/reader' if subfolder is None else RESULTS_PATH + f'/reader/{subfolder}/'
    field_list = []
    file_list = []

    for dataset in datasets:
        for model in models:
            for retriever in retrievers:
                for cond in conditions:
                    extra_fields = {'dataset': dataset,
                                    'model': model,
                                    'retriever': retriever,
                                    'condition': cond}
                    file_iter = pathlib.Path(base_path).rglob(f'{dataset}-{model}*{retriever}*noise-{cond}*mean-ci.score')
                    for f in file_iter:
                        fstr = str(f)
                        file_list.append(fstr)
                        field_list.append(extra_fields)

                # need gold alone and retriever alone
                file_iter = pathlib.Path(base_path).rglob(f'{dataset}-{model}-*-{retriever}*mean-ci.score')
                extra_fields = {
                    'dataset': dataset,
                    'model': model,
                    'retriever': retriever,
                    'condition': 0
                }
                for f in file_iter:
                    fstr = str(f)
                    if '-noise-' not in fstr:
                        file_list.append(fstr)
                        field_list.append(extra_fields)

    return file_list, field_list


def compile_metric_df(filenames, extra_fields=None, nested=False):
    """
    Utility to concatenate the data fields from all files in `filenames`, in addition to the `extra_fields` specified for each file.

    If you are doing this with `perquery.score` files, set `nested=True`. This is because the score file has some fields with single values, and some fields with a list of values. These need to be handled specially.
    """
    if extra_fields:
        assert len(filenames) == len(extra_fields), f"Expected additional field dicts to be in a list of len {len(filenames)}"
    data_list = []
    for i, f in enumerate(filenames):
        if nested:
            file_data = pd.read_json(f)
            for k, v in extra_fields[i].items():
                file_data[k] = v
        else:
            mdict = load_json(f)
            if extra_fields:
                mdict.update(extra_fields[i])
            file_data = pd.Series(mdict)
        data_list.append(file_data)
    if nested:
        return pd.concat(data_list)
    else:
        return pd.DataFrame(data_list)

def get_ci_df(datasets, models, cond_list, subfolder):
    file_list, field_list = load_metric_ci_files(datasets, models, cond_list, subfolder)
    df = compile_metric_df(file_list, field_list)
    return df

def remove_errorbar_in_legend(ax):
    """
    Removes the error bars from the legend.
    To add the legend based on this output, call `matplotlib.pyplot.legend(handles, labels)`
    """
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, matplotlib.container.ErrorbarContainer) else h for h in handles]
    return handles, labels 
