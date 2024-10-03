#!/bin/bash

export DATA_PATH=/export/data/aleto/rag_data
export INDEX_PATH=$DATA_PATH/indices
export VEC_PATH=$DATA_PATH/vectors
export DATASET_PATH=$DATA_PATH/datasets
export RESULTS_PATH=results
export COLBERT_MODEL_PATH=/export/data/vyvo/models/

echo "Environment variable DATA_PATH set to:" $DATA_PATH
echo "Environment variable DATASET_PATH set to:" $DATASET_PATH
echo "Environment variable INDEX_PATH set to:" $INDEX_PATH
echo "Environment variable VEC_PATH set to:" $VEC_PATH
echo "Environment variable RESULTS_PATH set to:" $RESULTS_PATH
echo "Environment variable COLBERT_MODEL_PATH set to:" $COLBERT_MODEL_PATH

# Now also set the pythonpath to search for the correct module files
current_path=$(pwd)
while [[ ${current_path: -4} != "alce" ]]; do
    current_path=$(dirname $current_path)
done
export PYTHONPATH=$current_path

echo "PYTHONPATH set to: " $PYTHONPATH
