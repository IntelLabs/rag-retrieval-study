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
Retrieval with dense text embeddings (e.g. the [BGE-1.5 embeddings[(https://huggingface.co/BAAI/bge-base-en-v1.5/tree/main)) is performed with Intel's Scalable Vector Search library. You can install it directly from pip by running `pip install scalable-vs`. It is imported into Python by `import svs`.

Alternatively, you can make more system-specific install configurations by following the [documentation here](https://intel.github.io/ScalableVectorSearch/).

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

Use the ```--not_par_level``` flag for asqa, where the gold metadata is not separated into document-level and paragraph-level ids.

To generate 10 noisy docs in each percentile of neighbors for each query, add the ```--noise_experiment``` tag. Note that this is only implemented for dense retrieval and has only been tested for asqa. 


### Reader

There are existing config files in reader/configs or you can create your own. You may also override arguments in the config file with command line arguments or choose not to use config files and specify everything
via command line arguments. 

```bash
python reader/run.py --config {config_name}
```

### Noise experiments
The process for performing experiments with adding noisy documents to gold and retrieved documents in the interest of replicating performance gains observed in [The Power of Noise](https://arxiv.org/abs/2401.14887) is outlined here. 

## Noise percentile experiments
Using the ```--noise_experiment``` tag in the retrieval step described in [Retriever evaluation](#retriever-evaluation) results in 10 noisy docs in each percentile of neighbors for each query. This is obtained by retrieving all documents for the query, resulting in an ordered list from most similar to least similar to the query. This is divided into ten equal bins. Random documents from each bin are exported to a noise file. This is implemented in `retriever/ret_utils.py`. For each resulting noise file, run:

```bash
python3 reader/run.py --config {config_name} --noise_file {noise_name}
```
By default, the noisy documents will be added to the prompt after the retrieved or gold documents. To switch this order, use the ```--noise_first``` flag. You can switch between adding noise to the gold and retrieved documents by changing the `config_name`. 


## First 100 neighbors experiments
To perform experiments with adding nearer neighbors to the gold and retrieved results run default retrieval to obtain an `eval_file`, then create new noise files for retrieved results 5-10 and 95-100 (for each query) by running: 

```bash
python3 preprocessing/sample_retrieved_neighbors.py --f {eval_file} --d {dataset_name}
```

Note that gold documents are omitted from this set. You can then run experiments with the resulting noise files (as outlined above): 

```bash
python3 reader/run.py --config {config_name} --noise_file {noise_name}
```

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
