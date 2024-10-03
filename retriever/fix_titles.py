import datasets
import json
import os
import tqdm


DATA_PATH = os.environ.get('DATA_PATH')
DATASET_PATH = os.environ.get('DATASET_PATH')

doc_dataset = "dpr_wiki"
in_path = 'asqa_retrieval_ann-recall0.9_bge-base-dense.json'
out_path = 'asqa_retrieval_ann-recall0.9_bge-base-dense-wtitles.json'

with open(os.path.join(DATA_PATH, in_path), 'r') as f:
    data = json.load(f)
print(f"Loaded {DATA_PATH}/{in_path}")

doc_titles = datasets.load_dataset("json", data_files=os.path.join(DATASET_PATH, doc_dataset, "id2title.jsonl"))
doc_titles = doc_titles["train"].to_dict()
print(f"Title dictionary created")
title_ids, titles = doc_titles['id'], doc_titles['title']
title_dict = {str(id): title for id, title in zip(title_ids, titles) if title is not None}
del doc_titles

for query in tqdm.tqdm(data):
    for doc in query['docs']:
        title = title_dict[doc['id']]
        if title is None:
            raise ValueError(f"Couldn't find title for {doc['id']}")
        doc['title'] = title

from file_utils import save_json

save_json(data, os.path.join(DATA_PATH, out_path))
print(f"Saved JSON to {out_path}")
