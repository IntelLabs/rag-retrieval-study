# Toward Optimal Search and Retrieval for RAG
This is a research code repository associated with the paper "Toward Optimal Search and Retrieval for RAG."

## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Code Structure](#code-structure)
  - [Retriever](#retriever)


## Requirements

Full details of the package versions we used for the latest experiments are in the `conda_env.yaml` file.
Details on which packages are required for each of the retrievers can be found in the [Retriever](#retriever) section. You will also need the HuggingFace `datasets` library and the `sentence-transformers` library to process the text data and embed it into vectors.

To run the reader portion of the RAG pipeline, i.e. LLM inference, please install the latest versions of PyTorch (`torch`), HuggingFace Transformers (`transformers`), HuggingFace Accelerate (`accelerate`). You will also need `nltk`.

To calculate evaluation scores for LLM outputs, you will also need `rouge-score`, and `scipy`.

## Code Structure
* `setup`: directory containing script for setting env variables
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

## Setup

### Download Data

To download ASQA and QAMPARI datasets, as well as the DPR wikipedia snapshot used for retrieved documents, please refer to the original [ALCE repository](https://github.com/princeton-nlp/ALCE). After downloading this data, create `asqa`, `qampari`, and `dpr_wiki` subdirectories in the location specified by the `DATASET_PATH` environment variable. Place one (it doesn't matter which) corresponding .json eval file in the `asqa` and `qampari` directories, respectively. Rename these files `raw.json`. Rename the downloaded dpr wikipedia dump `raw.tsv` and place it in the `dpr_wiki` subdirectory. Rename the oracle files included in the ALCE data `asqa_gold.json` and `qampari_gold.json`. Move them to the location specified by the `DATA_PATH` environment variable. Finally, the renamed ALCE .json files and DPR wikipedia .tsv file can be converted to the formats needed for running retrieval with SVS (Scalable Vector Search) by running:


```bash
python preprocessing/alce/convert_alce_dense.py --dataset {asqa/qampari}
```
Note that the DPR wikipedia split will need to be vectorized prior to running retrieval with SVS. 

To preprocess these files for use with ColBERT, simply run: 

```bash
python preprocessing/alce/convert_alce_colbert.py --dataset {asqa/qampari}
```

For the NQ dataset and the KILT Wikipedia corpus that supports it, you may follow the dataset download instructions as provided by original [RAGGED repository](https://github.com/neulab/ragged). This includes downloading the preprocessed corpus on [HuggingFace](https://huggingface.co/datasets/jenhsia/ragged). The original repository also provides tools to convert the data for use with ColBERT.

To preprocess the files for use with our dense retrieval code using SVS, run `preprocessing/convert_nq_dense.py` with the appropriate input arguments.

### Set Paths
Before getting started, you must fill in the path variables `setup/set_paths.sh` for your environment

```bash
export DATA_PATH=  # directory containing all preprocessed eval files
export INDEX_PATH=$DATA_PATH/indices  # directory to save indices for search/retrieval with SVS
export VEC_PATH=$DATA_PATH/vectors  # path to document vectors for search/retrieval with SVS
export DATASET_PATH=  # directory containing subdirectories (labelled with dataset name) containing raw downloaded data
export RESULTS_PATH=  # location to save output from retriever and reader eval
export COLBERT_MODEL_PATH=  # location where colbert model has been downloaded
```

then run with 

```bash
source setup/set_paths.sh
```

## Retriever

You can complete the passage retrieval step with the following command:
```bash
python retriever/run.py --config {config_name}
```

There are existing config files in retriever/configs or you can create your own. You may also override arguments in the config file with command line arguments or choose not to use config files and specify everything via command line arguments. 

There are additional packages required for the retrieval steps. The packages differ for each type of retriever.

### Dense embedding retrieval with SVS (Scalable Vector Search)
Retrieval with dense text embeddings (e.g. the [BGE-1.5 embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5/tree/main)) is performed with Intel's Scalable Vector Search library. You can install it directly from pip by running `pip install scalable-vs`. It is imported into Python by `import svs`.

Alternatively, you can make more system-specific install configurations by following the [documentation here](https://intel.github.io/ScalableVectorSearch/).

We have implemented similarity-based retrieval with either exact search or approximate nearest neighbor (ANN) search. Retriever configuration files for exact search are titled `dense_{DATASET}.yaml`, while approximate search parameters can be set in configuration files titled `dense_ann_{DATASET}.yaml`. Several of the parameters for building the ANN search graph can be modified to alter the search performance, but we have provided configurations that work well for those datasets.

#### Tuning search recall
For some experiments, we have tuned the ANN search to achieve a specific accuracy compared to exact search. Therefore we have included a `preprocessing/create_ground_truth_calibration.py` script to save the results of exact search on a subset of the data.

#### Setting gold document recall
For some experiments, we manipulated the set of context documents to achieve an exact number for average retrieval recall across the whole dataset of queries. This can be run with `preprocessing/set_gold_recall.py`.

### ColBERT-2.0 retriever -- very slight modifications from v.0.2.20
The faiss package is required to run the ColBERT retriever.

We have added a slightly modified version of the [main ColBERT repository](https://github.com/stanford-futuredata/ColBERT), version 0.2.20 (commit b7352c2), to our repository. The only modifications are made to the `colbert/indexing/collection_indexer.py` file in the `_build_ivf` function, and they do not alter the functional output of the code -- they simply enable running the code with very large corpora.

Details on these changes: We ran into OOM errors when using the original code. This occurred on line 467 of the `_build_ivf` function, which attempts to sort a pytorch tensor containing a mapping from the vector embedding index to the IVF centroids/partitions. Simply using a numpy array and numpy sort fixes this OOM error. We also replace the call to torch.bincount with np.bincount before converting the numpy arrays to torch tensors to maintain continuity with the rest of the code base.

### Retriever evaluation
To evaluate retrieval results use the following command:
```bash
python retriever/eval.py --eval_file {eval_filename} --not_par_level
```

Use the ```--not_par_level``` flag for asqa, where the gold metadata is not separated into document-level and paragraph-level ids.

To generate 10 noisy docs in each percentile of neighbors for each query, add the ```--noise_experiment``` tag. Note that this is only implemented for dense retrieval and has only been tested for asqa. 


## Reader

There are existing config files in reader/configs or you can create your own. You may also override arguments in the config file with command line arguments or choose not to use config files and specify everything via command line arguments. 

```bash
python reader/run.py --config {config_name}
```

Bash files for looping over various numbers of documents included in the prompt and evaluating the results can be found in `runners/ndoc_asqa_mistral_reader.sh` and `runners/ndoc_asqa_mistral_eval_looper.sh`. 

### Noise experiments
The process for performing experiments with adding noisy documents to gold and retrieved documents in the interest of replicating performance gains observed in [The Power of Noise](https://arxiv.org/abs/2401.14887) is outlined here. 

## Noise percentile experiments
Using the ```--noise_experiment``` tag in the retrieval step described in [Retriever evaluation](#retriever-evaluation) results in 10 noisy docs in each percentile of neighbors for each query. This is obtained by retrieving all documents for the query, resulting in an ordered list from most similar to least similar to the query. This is divided into ten equal bins. Random documents from each bin are exported to a noise file. This is implemented in `retriever/ret_utils.py`. For each resulting noise file, run:

```bash
python3 reader/run.py --config {config_name} --noise_file {noise_name}
```
By default, the noisy documents will be added to the prompt after the retrieved or gold documents. To switch this order, use the ```--noise_first``` flag. You can switch between adding noise to the gold and retrieved documents by changing the `config_name`. 

Bash files for running and evaluating this experiment can be found at `runners/noise_percentile_asqa_mistral_gold_reader.sh` and `runners/noise_percentile_asqa_mistral_gold_eval.sh`. 


## First 100 neighbors experiments
To perform experiments with adding nearer neighbors to the gold and retrieved results run default retrieval to obtain an `eval_file`, then create new noise files for retrieved results 5-10 and 95-100 (for each query) by running: 

```bash
python3 preprocessing/sample_retrieved_neighbors.py --f {eval_file} --d {dataset_name}
```

Note that gold documents are omitted from this set. You can then run experiments with the resulting noise files (as outlined above): 

```bash
python3 reader/run.py --config {config_name} --noise_file {noise_name}
```

Bash files for running and evaluating this experiment can be found at `runners/first100_asqa_mistral_bge-base_reader.sh` and `runners/first100_asqa_mistral_bge-base_eval.sh`.

### Reader evaluation

Accuracy on the QA task is implemented in run/eval.py. The evaluation code contains copies of functions from two RAG papers that previously used these datasets ([ALCE](https://github.com/princeton-nlp/ALCE) and [RAGGED](https://github.com/neulab/ragged)).

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

## Disclaimer
This “research quality code” is provided by Intel “As Is” without any express 
or implied warranty of any kind. Intel does not warrant or assume 
responsibility for the accuracy or completeness of any information, text, 
graphics, links or other items within the code. A thorough security review 
has not been performed on this code. Additionally, this repository will not 
be actively maintained and as such may contain components that are out of 
date, or contain known security vulnerabilities.  Proceed with caution.
