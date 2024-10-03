import os
import json
from tqdm import tqdm
from file_utils import load_json, save_json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Updates key values in nq query files and kilt dataset for consistency across datasets
"""

"""
Updates key values in nq query files and kilt dataset for consistency across datasets
"""

def main():

    DATA_PATH = os.environ.get("DATA_PATH")
    input_file = os.path.join(DATA_PATH, "nq_eval-test-svs-copy.json")

    # # In nq query .json, renames 'input' key 'question'
    # input_file = os.path.join(DATA_PATH, "nq_eval.json")

    # with open(input_file) as f:
    #     input_data = json.load(f)

    # clean_data = []
    # for entry in input_data:
    #     entry['question'] = entry['input']
    #     del entry['input']
    #     clean_data.append(entry)

    
    # with open(input_file, "w") as f:
    #         json.dump(clean_data, f, indent=4)


    # # In kilt .jsonl file, renames "contents" key "text"
    # DATASET_PATH = os.environ.get("DATASET_PATH")
    # print("Reading original file")
    # input_file = os.path.join(DATASET_PATH, 'kilt_wikipedia', 'docs.jsonl')
    # with open(input_file, 'r') as json_file:
    #     json_list = list(json_file)

    # clean_data = []
    # print("Changing key value")
    # for d in tqdm(json_list):
    #     d = json.loads(d)
    #     d['text'] = d['contents']
    #     del d['contents']
    #     clean_data.append(d)

    # print("Printing to file")
    # with open(input_file, 'w') as jsonl_output:
    #     for entry in tqdm(clean_data):
    #         json.dump(entry, jsonl_output)
    #         jsonl_output.write('\n')


    # adds gold answer to nq eval file
    input_data = load_json(input_file)

    clean_data = []
    for d in tqdm(input_data):
        d["answer"] = d["output"]["answer_set"][0]
        clean_data.append(d)

    input_file = os.path.join(DATA_PATH, "nq_eval-test-svs.json")
    save_json(clean_data, input_file, logger)

   


if __name__ == "__main__": 
    main()