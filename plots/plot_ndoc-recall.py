import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_utils

def main():
    plot_utils.set_plot_params()
    datasets = ['asqa', 'nq', 'qampari']
    retrievers = ['bge-base', 'colbert']
    models = ['Llama', 'Mistral']
    k_vals = [1, 2, 3, 4, 5, 10, 20, 50, 100]

    acc_keys = {
        "nq": "ragged_substring_match",
        "qampari": "qampari_rec_top5"
    }
    ret_colors = {
        "bge-base": "blue",
        "colbert": "orange"
    }

    # grab the files and generate data frames
    print("Loading data from ndoc experiments...")
    file_list, field_list = plot_utils.load_metric_ndoc_files(
        datasets, models, retrievers, conditions=k_vals
    )

    df = plot_utils.compile_metric_df(file_list, extra_fields=field_list, nested=False)
    df.to_csv('ndoc_reader_recall.csv')

    # now generate the plots!
    df_datasets = ['ASQA', 'NQ', 'QAMPARI']  # order that I want to plot the datasets
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # a plot for each retriever/model combination
    print('\n\nGenerating plots...')
    for model in models:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.2))
        for ax_num, dataset in enumerate(datasets):
            axh_plot = False
            for retriever in retrievers:
                df_filtered = df[(df['model'] == model) & (df['retriever'] == retriever) & (df['dataset'] == dataset)]
                if len(df_filtered) == 0:
                    print(f"Could not find data for {dataset}, {model}, {retriever}, skipping...")
                    continue
                print(f"Plotting {model}, {retriever}, {dataset}...")
                if not axh_plot:  # plot the no-context and gold lines
                    no_context_acc = df[(df['model'] == model) & (df['dataset'] == dataset) & (df['condition'] == 'no-context')][acc_keys[dataset]].values[0]
                    ax[ax_num].axhline(y=no_context_acc, color='red', linestyle='--', label='No Context')
                    gold_acc = df[(df['model'] == model) & (df['dataset'] == dataset) & (df['condition'] == 'gold')][acc_keys[dataset]].values[0]
                    ax[ax_num].axhline(y=gold_acc, color='green', linestyle='--', label='Gold')
                    axh_plot = True
                print(df_filtered['condition'])
                print(df_filtered[acc_keys[dataset]])
                ax[ax_num].plot(df_filtered['condition'], df_filtered[acc_keys[dataset]], color=ret_colors[retriever], label=retriever)
                ax[ax_num].scatter(df_filtered['condition'], df_filtered[acc_keys[dataset]], color=ret_colors[retriever])

                acc_label = 'EM Recall'
                if dataset == 'qampari':
                    acc_label += " -5"
                ax[ax_num].set_ylabel(acc_label, fontsize=16)
                ax[ax_num].set_xlabel('k', fontsize=16)
                ax[ax_num].tick_params(labelsize=12)
                ax[ax_num].set_title(f'{dataset.upper()}', fontsize=18)

                if ax_num == 2:  # only want one legend
                    ax[ax_num].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=12)
                else:  # remove the legend for the other plots
                    ax[ax_num].legend().remove()
    

        # Save the figure to an output file
        fig.tight_layout()
        figname = f'plots/ndoc-reader-acc-{model}.png'
        plt.savefig(figname)



if __name__ == "__main__":
    main()