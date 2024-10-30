import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_utils
import seaborn as sns


def main():
    plot_utils.set_plot_params()
    colors = sns.color_palette()

    datasets = ['asqa', 'qampari', 'nq', 'nq-nocite']
    retrievers = ['bge-base', 'colbert']
    models = ['Llama', 'Mistral']
    k_vals = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    shared_y = False

    subplot_indices = {
        'asqa': [0, 0],
        'qampari': [0, 1],
        'nq': [1, 0],
        'nq-nocite': [1, 1]
    }

    # grab the files and generate data frames
    print("Loading data from ndoc experiments...")
    file_list, field_list = plot_utils.load_metric_ndoc_files(
        datasets, models, retrievers, conditions=k_vals
    )

    df = plot_utils.compile_metric_df(file_list, extra_fields=field_list, nested=False)
    print(df.describe())

    # a plot for each retriever/model combination
    print('\n\nGenerating plots...')
    for model in models:
        fig, ax = plt.subplots(2, 2, figsize=(5.5, 4), width_ratios=[1, 1])
        for dataset in datasets:
            axh_plotted = False  # flag to plot the no-context and gold lines (only once per dataset)
            for retriever in retrievers:
                df_filtered = df[(df['model'] == model) & (df['retriever'] == retriever) & (df['dataset'] == dataset)]
                if len(df_filtered) == 0:
                    print(f"Could not find data for {dataset}, {model}, {retriever}, skipping...")
                    continue
                print(f"Plotting {model}, {retriever}, {dataset}...")
                ax_x = subplot_indices[dataset][0]
                ax_y = subplot_indices[dataset][1]
                if not axh_plotted:  # plot the no-context and gold lines
                    no_context_acc = df[(df['model'] == model) & (df['dataset'] == dataset) & (df['condition'] == 'no-context')]["em_rec_mean"].values[0]
                    ax[ax_x][ax_y].axhline(y=no_context_acc, color=colors[3], linestyle='--', label='No Context')
                    gold_acc = df[(df['model'] == model) & (df['dataset'] == dataset) & (df['condition'] == 'gold')]["em_rec_mean"].values[0]
                    ax[ax_x][ax_y].axhline(y=gold_acc, color=colors[2], linestyle='--', label='Gold')
                    axh_plotted = True

                # plot em rec mean with error bars
                x = np.array(df_filtered['condition'])
                m = np.array(df_filtered['em_rec_mean'])
                yerr = np.stack([df_filtered['em_rec_ci_lower'], df_filtered['em_rec_ci_upper']], axis=0)
                ax[ax_x][ax_y].errorbar(x, m, yerr=abs(yerr - m), label=retriever)

                if dataset == 'nq-nocite':
                    ax[ax_x][ax_y].set_title("NQ (No Citations)")
                else:
                    ax[ax_x][ax_y].set_title(f'{dataset.upper()}')

                acc_label = 'EM Recall'
                if ax_y == 0:  # only show y-axis label on the left column
                    ax[ax_x][ax_y].set_ylabel(acc_label)

                if ax_x == 1:  # only show x-axis label on the bottom row
                    ax[ax_x][ax_y].set_xlabel('k')

                if shared_y:
                    ax[ax_x][ax_y].set_ylim(0, 90)

                if 'nq' in dataset:
                    ax[ax_x][ax_y].set_ylim(0, 90)

        # legend with only labels from first plot (top left)
        handles, labels = ax[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4)

        # Save the figure to an output file
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.17)  # fix overlap with x axis label

        figname = f'plots/ndoc-reader-acc-{model}.png'
        if shared_y:
            figname = f'plots/ndoc-reader-acc-{model}-shared-y.png'
        plt.savefig(figname)


if __name__ == "__main__":
    main()
