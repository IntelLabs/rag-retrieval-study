import numpy as np
import matplotlib.pyplot as plt
import plot_utils
import seaborn as sns

def main():

    plot_utils.set_plot_params()
    colors = sns.color_palette()

    datasets = ['asqa']
    retrievers = ['bge-base', 'gold']
    models = ['Mistral']
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # grab the files and generate data frames
    print("Loading data from ndoc experiments...")
    file_list, field_list = plot_utils.load_metric_noise_files(
        datasets, models, retrievers, conditions=percentiles, subfolder="noise"
    )
    df = plot_utils.compile_metric_df(file_list, field_list)

    print("Generating plots...")
    for dataset in datasets:
        for model in models:
            fig, ax = plt.subplots(figsize=(5, 2))
            df_dataset = df[df['dataset'] == dataset]

            for idx, retriever in enumerate(retrievers):

                df_ret = df_dataset[df_dataset['retriever'] == retriever]
                ret_only = df_ret[df_ret['condition'] == 0]  # retriever only
                df_ret = df_ret[df_ret['condition'] != 0]  # remove retriever only

                # plot the data
                x = np.array(df_ret['condition'])
                m = np.array(df_ret['em_rec_mean'])
                yerr = np.stack([df_ret['em_rec_ci_lower'], df_ret['em_rec_ci_upper']], axis=0)
                ax.plot(x, m, label=f"{retriever} + noise", color=colors[idx])
                ax.errorbar(x, m, yerr=abs(yerr - m), color=colors[idx])
                ax.axhline(ret_only['em_rec_mean'].values[0], linestyle='--', label=retriever, color=colors[idx])


            plt.ylabel("EM Recall")
            plt.xlabel("Noise Percentile")
            plt.title(f"{dataset.upper()} ({model})")
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            plt.ylim([20, 55])
            plt.tight_layout()
            fig.savefig(f"plots/rand_percentile_recall_{dataset}_{model}.png", dpi=300)


if __name__ == "__main__":
    main()