import os
from file_utils import load_json, save_json
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    DATA_PATH = os.environ.get("DATA_PATH")
    data_path = os.path.join(DATA_PATH, "nq_retrieval-colbert_old-id.json")
    retrieved_data = load_json(data_path, logger=logger)
    output_file = os.path.join(DATA_PATH, "nq_retrieval-colbert.json")

    gold_file = os.path.join(DATA_PATH, "nq_gold.json")
    gold_data = load_json(gold_file)

    output_data = []
    for ret, gold in zip(retrieved_data, gold_data):
        assert ret['id'] == gold['id'], "Retrieved and gold entries don't match!"
        ret['output'] = gold['output']
        output_data.append(ret)
    
    save_json(output_data, output_file, logger=logger)


if __name__ == "__main__": 
    main()
