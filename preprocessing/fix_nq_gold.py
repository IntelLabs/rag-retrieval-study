import os
from file_utils import load_json, save_json
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    DATA_PATH = os.environ.get("DATA_PATH")
    result_folder = os.path.join(os.environ.get("RESULTS_PATH"), 'reader', 'nq_ndoc')
    paths = ['/llama/colbert/', '/mistral/colbert/']

    gold_file = os.path.join(DATA_PATH, "nq_gold.json")
    gold_data = load_json(gold_file)

    for pdir in paths:
        result_files = os.listdir(result_folder + pdir)
        for data_file in result_files:
            data_path = f"{result_folder}{pdir}/{data_file}"
            output_file = f"{result_folder}{pdir}/{data_file[:-5]}_fixedgold.json"
            reader_data = load_json(data_path, logger=logger)
            rd_data = reader_data['data']
            output_data = []
            for rd, gold in zip(rd_data, gold_data):
                assert rd['id'] == gold['id'], "Retrieved and gold entries don't match!"
                rd['output'] = gold['output']
                output_data.append(rd)
            reader_data['data'] = output_data
            save_json(reader_data, output_file, logger=logger)
            logger.info(f"Wrote to {output_file}")


if __name__ == "__main__": 
    main()
