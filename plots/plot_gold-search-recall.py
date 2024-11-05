# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_utils


plot_utils.set_plot_params()
datasets = ['asqa', 'nq']
models = ['Llama', 'Mistral']

# Grab the files and generate data frames
print("Loading data from gold-recall experiments...")
file_list, field_list = plot_utils.load_metric_ci_files(datasets, models, conditions=[0.5, 0.7, 0.9, 1.0], subfolder="gold-recall")
goldrec_df = plot_utils.compile_metric_df(file_list, field_list)
#print(goldrec_df)
print("Loading data from search-recall experiments...")
file_list, field_list = plot_utils.load_metric_ci_files(datasets, models, conditions=[0.7, 0.9, 0.95], subfolder="search-recall")
searchrec_df = plot_utils.compile_metric_df(file_list, field_list)

# Now generate the plot itself!
df_datasets = ['ASQA', 'NQ', 'NQ-nocite']  # order that I want to plot the datasets
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for model in models:
    #f, ax = plt.subplots(1, 2, figsize=(4.25, 1.75), width_ratios=[2, 1])
    f, ax = plt.subplots(1, 2, figsize=(5.5, 2), width_ratios=[2, 1])
    p0 = goldrec_df[goldrec_df['model'] == model]
    for i, d in enumerate(df_datasets):
        this_plot = p0[p0['dataset'] == d]
        x = np.array(this_plot['condition'])
        m = np.array(this_plot['em_rec_mean'])
        yerr = np.stack([this_plot['em_rec_ci_lower'], this_plot['em_rec_ci_upper']], axis=0)
        ax[0].errorbar(x, m, yerr=abs(yerr - m), label=d)
        # Plot shaded bar for recall = 1.0
#        print(p0)
        rdf = p0[(p0['dataset'] == d) & (p0['condition'] == 1.0)]
        if len(rdf) == 0:
            print(f"Could not find data for {model}, {d}, skipping...")
            continue
        ax[0].axhspan(rdf['em_rec_ci_lower'].item(), rdf['em_rec_ci_upper'].item(), color=colors[i, :3], alpha=0.15)
        ax[1].axhspan(rdf['em_rec_ci_lower'].item(), rdf['em_rec_ci_upper'].item(), color=colors[i, :3], alpha=0.15)
    ax[0].set_ylim([20, 90])
    ax[0].set_xlabel('Gold Document Recall')
    ax[0].set_ylabel('EM Recall')
    ax[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax[0].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax[0].grid(which='minor', alpha=0.5)
    ax[0].invert_xaxis()
    # Add legend
    #ax[0].legend(loc='lower left', ncols=3, columnspacing=0.5, frameon=False, bbox_to_anchor=(-0.02, -0.02), handletextpad=0.5)
    ax[0].legend(loc='lower left', ncols=3, frameon=True)
    
    p1 = searchrec_df[searchrec_df['model'] == model]
    for d in df_datasets:
        this_plot = p1[p1['dataset'] == d]
        x = np.array(this_plot['condition'])
        m = np.array(this_plot['em_rec_mean'])
        yerr = np.stack([this_plot['em_rec_ci_lower'], this_plot['em_rec_ci_upper']], axis=0)
        ax[1].errorbar(x, m, yerr=abs(yerr - m), label=d)
    ax[1].set_ylim([20, 90])
    ax[1].set_xlabel('Search Recall@10')
    ax[1].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax[1].grid(which='minor', alpha=0.5)
    ax[1].invert_xaxis()
    plt.tight_layout()

    # Save the figure to an output file
    figname = f'plots/gold-search-recall-{model}.png'
    plt.savefig(figname)
    print(f'Saved {figname}!')
    plt.close(f)
    del f, ax

