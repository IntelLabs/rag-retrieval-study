import argparse
from file_utils import load_json, save_json
import os
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESULTS_PATH = os.environ.get("RESULTS_PATH")

"""
 Creates a sample output file with examples removed from prompt and generated output
"""

def main(args):

    args.f = os.path.join(RESULTS_PATH, 'reader', args.f)
    results_raw = load_json(
        args.f,
        logger
    )
    results_raw = results_raw['data']
    clean_results = []

    for entry in results_raw:

        clean_entry = {}
        clean_entry['id'] = entry['id']
        clean_entry['prompt'] = "Instruction" + entry["prompt"].split("Instruction")[3]
        clean_entry['generated_output'] = entry['generated_output']

        clean_results.append(clean_entry)


    outfile = args.f.replace(".json.score", "") + "-sample.json"
    save_json(clean_results, outfile, logger)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Name of results file to sample")
    args = parser.parse_args()
    main(args) 