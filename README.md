# ALCE-RAGGED Integrated Repo


## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Code Structure](#code-structure)
  - [Retriever](#retriever)


## Requirements

Please install the latest versions of PyTorch (`torch`), HuggingFace Transformers (`transformers`), HuggingFace Accelerate (`accelerate`), and the OpenAI API package (`openai`). This codebase is tested on 
`torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, `accelerate==0.17.1`, and `openai==0.27.4` with Python 3.9.7.
(add to this list to reflect ragged requirements) 


## Code Structure
* `setup`: directory containing scripts for downloading data and setting env variables
* `retriever`: directory containing scripts and files for retrieval and retriever eval
  * `configs`: config files for different retrieval settings
  * `index.py`: index datasets before or during retrieval step
  * `run.py`: retrieval, pass config file
  * `eval.py`:   
* `reader`: directory containing scripts and files for generation and generation eval
  * `configs`: config files for different generation settings
  * `prompts`: folder that contains all prompt files.
  * `run.py`: retrieval, pass config file
  * `eval.py`: eval file to evaluate generations.
* `tools/`: misc code (generate summaries/snippets, reranking, etc.)



## Retriever

You can complete the passage retrieval step with the following command:
```bash
python retriever/run.py --config {config_name}
```

There are existing config files in retriever/configs or you can create your own. You may also override arguments in the config file with command line arguments or choose not to use config files and specify everything via command line arguments. 

There are additional packages required for the retrieval steps. The packages differ for each type of retriever.

### Dense embedding retrieval with SVS (Scalable Vector Search)
Retrieval with dense text embeddings (e.g. the [BGE-1.5 embeddings[(link)) is performed with Intel's Scalable Vector Search library. This is available on pip or conda. [Documentation here](https://intel.github.io/ScalableVectorSearch/).
We have implemented similarity-based retrieval with either exact search or approximate nearest neighbor (ANN) search. Retriever configuration files for exact search are titled `dense_{DATASET}.yaml`, while approximate search parameters can be set in configuration files titled `dense_ann_{DATASET}.yaml`. Several of the parameters for building the ANN search graph can be modified to alter the search performance, but we have provided configurations that work well for those datasets.

For some experiments, we have tuned the ANN search to achieve a specific accuracy compared to exact search. Therefore we have included a `preprocessing/create_ground_truth_calibration.py` script to save the results of exact search on a subset of the data.

### ColBERT-2.0 retriever -- very slight modifications from v.0.2.20
The faiss package is required to run the ColBERT retriever.

We have added a slightly modified version of the [main ColBERT repository](https://github.com/stanford-futuredata/ColBERT), version 0.2.20 (commit b7352c2), to our repository. The only modifications are made to the `colbert/indexing/collection_indexer.py` file in the `_build_ivf` function, and they do not alter the functional output of the code -- they simply enable running the code with very large corpora.

Details on these changes: We ran into OOM errors when using the original code. This occurred on line 467 of the `_build_ivf` function, which attempts to sort a pytorch tensor containing a mapping from the vector embedding index to the IVF centroids/partitions. Simply using a numpy array and numpy sort fixes this OOM error. We also replace the call to torch.bincount with np.bincount before converting the numpy arrays to torch tensors to maintain continuity with the rest of the code base.


### Retriever evaluation
To evaluate retrieval results use the following command:
```bash
python retriever/eval.py --eval_file {eval_filename} --not_par_level
```

Use the --not_par_level flag for asqa, where the gold metadata is not separated into document-level and paragraph-level ids.

To generate 10 noisy docs in each percentile of neighbors for each query, add the ```--noise_experiment``` tag. Note that this is only implemented for dense retrieval and has only been tested for asqa. 


### Noise experiments
TODO: add overall statement about noise experiments and how to run them. specifics below
TODO: add info about retrieving ALL neighbors
TODO: add info about noise percentile bins in `ret_utils.py` and `run.py`
TODO: add info about evaluating closer neighbors with `preprocessing/sample_retrieved_neighbors.py` and `run.py`


## Reader

There are existing config files in reader/configs or you can create your own. You may also override arguments in the config file with command line arguments or choose not to use config files and specify everything
via command line arguments. 

```bash
python reader/run.py --config {config_name}
```

### Reader evaluation

ACLE evaluation is implemented in `run/eval.py`. The metrics are tailored to each dataset as outlined in the script. 

For ASQA and QAMPARI, use the following command
```bash
python reader/eval.py --f {result_file_name} --citations
```

For nq and bioasq, use the following command. Note that an option to run this eval without bert is offered because it can be somewhat time consuming. 
```bash
python reader/eval.py --f {result_file_name} --citations --no_bert
```

The evaluation result will be saved in `result/`, with the same name as the input and a suffix `.score`.

To generate per-k reader result plots with retrieval results on the same axis, run: 

```bash
python reader/plot_per_k.py --eval_file {dataset}-{model_name}-None-shot{}-ndoc*-42-{cite-}{retriever}.json.score --ret_file {dataset}_retrieval-{retriever}.json --ret_metric {top-k accuracy/precision@k/recall@k}
```









# Enabling Large Language Models to Generate Text with Citations

<p align="center"><img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/moose.png?raw=true" alt="ALCE" width="15%"><br>*: ALCE is pronounced as /elk/ as ALCE is the Latin word for elk (Europe) or moose (North America).
</p>



This repository contains the code and data for paper [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627). 
In this paper, we propose ALCE, a benchmark for **A**utomatic **L**LMs' **C**itation Evaluation. 
ALCE contains three datasets: ASQA, QAMPARI, and ELI5.
We provide automatic evaluation code of LLM generations around three dimensions: fluency, correctness, and citation quality. 
This repository also includes code to reproduce the baselines in our paper.



<img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/ALCE.png?raw=true" alt="ALCE" width="100%">




## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Code Structure](#code-structure)
  - [Reproducing Baselines](#reproducing-baselines)
  - [Evaluation](#evaluation)
  - [Human Evaluation](#human-evaluation)
  - [Bug or Questions](#bug-or-questions)
  - [Citation](#citation)


## Requirements

Please install the latest versions of PyTorch (`torch`), HuggingFace Transformers (`transformers`), HuggingFace Accelerate (`accelerate`), and the OpenAI API package (`openai`). This codebase is tested on 
`torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, `accelerate==0.17.1`, and `openai==0.27.4` with Python 3.9.7.

## Data

You can download datasets (along with retrieval results) by running the following command:

```bash
bash download_data.sh
```

All the data will be stored in `data/`. Our data included top-100 DPR/GTR retrieved results for ASQA and QAMPARI, and top-100 BM25 retrieved results for QAMPARI. We also provide reranked oracle retrieval results, where top-5 passages can achieve the same recall as the original top-100 recall.

### Retrieval

You can reproduce the passage retrieval step with the following command:
```bash
python retrieval.py --data {path/to/data} --retriever {bm25/gtr} --output_file {path/to/output}
```

There are additional packages required for the retrieval steps.
Specifically, you need to install `pyserini==0.21.0`(their github [repo](https://github.com/castorini/pyserini/tree/master) is helpful) and `sentence-transformers==2.2.2`.

For the BM25 retrieval over Common Crawl using Sphere, you must first download the index from the Sphere [repo](https://github.com/facebookresearch/Sphere), and set the environmental variable `BM25_SPHERE_PATH` to the path of the downloaded index.
Specifically, you can use the following command:
```bash
wget -P faiss_index https://dl.fbaipublicfiles.com/sphere/sphere_sparse_index.tar.gz
tar -xzvf faiss_index/sphere_sparse_index.tar.gz -C faiss_index
export BM25_SPHERE_PATH=$PWD/faiss_index
```
It's important to note that given the large size of the corpus, this step is extremely expensive and time-consuming. We found that larger CPU memory tends to help with the speed. 

For GTR, we first build an index using the DPR wikipedia snapshot, which you can obtain using the download script from the DPR [repo](https://github.com/facebookresearch/DPR), and then setting the environmental variable `DPR_WIKI_TSV` to the path of the tsv file.
Specifically, you can use the following command:
```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -xzvf psgs_w100.tsv.gz
export DPR_WIKI_TSV=$PWD/psgs_w100.tsv
```
Then, you want to set `GTR_EMB` to the path of the GTR embeddings of the Wikipedia corpus, and running the retrieval script for the first time will automatically build and save the index.
Building the dense index can be expensive for GPU memory (we use 80GB GPUs for this) and time-consuming; the entire index will take about 31GB.
If you find this step to be too expensive, you can also download it using:
```bash
wget https://huggingface.co/datasets/princeton-nlp/gtr-t5-xxl-wikipedia-psgs_w100-index/resolve/main/gtr_wikipedia_index.pkl
export GTR_EMB=$PWD/gtr_wikipedia_index.pkl
```

To reproduce the DPR retrieval, we refer the DPR [repo](https://github.com/facebookresearch/DPR), which we used the original DPR checkpoint trained on NQ.

## Code Structure

* `run.py`: run file to reproduce our baseline generations.
* `eval.py`: eval file to evaluate generations.
* `prompts`: folder that contains all prompt files.
* `configs/`: folder that contains all config files to reproduce baselines.
* `tools/`: misc code (generate summaries/snippets, reranking, etc.)


## Reproducing Baselines


You can reproduce baselines from our paper by 

```bash
python run.py --config configs/{config_name}
```

You can also overwrite any arguments in the config file or add new arguments simply through command line:
```
python run.py --config configs/{config_name} --seed 43 --model vicuna-13b
```

The naming of config files follow the rule of `{LLM}_{#demos and #passages}_{retriever}_{method}.yaml`. Method names include:
* `default` corresponds to the **Vanilla** model in our paper.
* `summary` corresponds to the **Summary** model.
* `extraction` corresponds to the **Snippet** model. 
* `interact_doc_id` corresponds to the **Interact** model.
* `interact_search` corresponds to the **InlineSearch** model.
* `closedbook` corresponds to the **ClosedBook** model.

Our code support both OpenAI API and offline HuggingFace models:

* For OpenAI models (for example, ChatGPT), you need to set the environment variable `OPENAI_API_KEY` and `OPENAI_ORG_ID`. If you are using the Azure OpenAI API, you need to set the environment variable of `OPENAI_API_KEY` and `OPENAI_API_BASE`. You also need to add the flag `--azure`. 
    * Note that in Azure OpenAI API, ChatGPT's name is different and you should set it by `--model gpt-35-turbo`. 
* For the open-source models, you should set the model name equal to the input of HuggingFace models' `.from_pretrained` method. This could either be a local directory (e.g. for the older LLaMA models) or a path to the HuggingFace hub. 

For detailed argument usage, please refer to `run.py`.

Model output along with gold answers and run configs will be stored in a json file in `result/`.


### Post-hoc citation

For closed-book models, one can use `post_hoc_cite.py` to add citations in a post-hoc manner (using GTR-large). To run post-hoc citation, execute
```bash
python post_hoc_cite.py --f result/{RESULT JSON FILE NAME} --external_docs data/{CORRESPONDING DATA}
```

The output file with post-hoc citations will be stored in `result/`, with a suffix `post_hoc_cite.gtr-t5-large-external`.

## Evaluation

ACLE evaluation is implemented in `eval.py`. 

For ASQA, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --qa --mauve
```

For QAMPARI, use the following command
```bash
python eval.py --f {path/to/result/file} --citations
```

For ELI5, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve
```

The evaluation result will be saved in `result/`, with the same name as the input and a suffix `.score`.

## Human Evaluation

The results from our human evaluation (Section 6) are located under the directory [`human_eval`](human_eval). 
Both the data and the analysis are available, please refer to the directory for details. 

## Bug or Questions?

If you have any questions related to the code or the paper, feel free to email Tianyu (`tianyug@cs.princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please cite our paper if you use ALCE in your work:

```bibtex
@inproceedings{gao2023enabling,
   title={Enabling Large Language Models to Generate Text with Citations},
   author={Gao, Tianyu and Yen, Howard and Yu, Jiatong and Chen, Danqi},
   year={2023},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
}
```
