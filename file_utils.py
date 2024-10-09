import json
import pickle

def save_json(data, filename, logger=None):
    if logger:
        logger.info(f'writing to {filename}')
    # assert filename.endswith("json"), "file provided to save_json does not end with .json extension. Please recheck!"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)

def load_json(filename, logger=None, sort_by_id = False):
    if logger:
        logger.info(f'loading from {filename}')
    assert filename.endswith("json") or filename.endswith(".score"), "file provided to load_json does not end with .json extension. Please recheck!"
    data = json.load(open(filename))
    if sort_by_id:
        for d in data:
            d["id"] = str(d["id"])
        return sorted(data, key=lambda x: x['id'])
    return data

def save_pickle(data, filename, logger=None):
    if logger:
        logger.info(f'writing to {filename}')
    with open(filename, 'wb') as ftmp:
        pickle.dump(data, ftmp)

def load_pickle(filename, logger=None):
    if logger:
        logger.info(f'loading from {filename}')
    with open(filename, 'rb') as ftmp:
        data = pickle.load(ftmp)
    return data

def load_jsonl(filename, sort_by_id = True, logger=None):
    if logger:
        logger.info(f'loading from {filename}')
    data = []
    with open(filename, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    if sort_by_id:
        for d in data:
            d["id"] = str(d["id"])
        return sorted(data, key=lambda x: x['id'])
    return data

def save_jsonl(data, filename, logger=None):
    if logger:
        logger.info(f'writing to {filename}')
    with open(filename, "w") as outfile:
        for idx, element in enumerate(data):
            json.dump(element, outfile)
            outfile.write("\n")
