import numpy as np
import pickle
import pysvs


gt_file = '/export/data/aleto/rag_data/vamana_calib_data/asqa_bge-base-dense_ground_truth.npy'
gt_qi_file = '/export/data/aleto/rag_data/vamana_calib_data/asqa_bge-base-dense_query_inds.npy'
ret_pkl = 'dpr_wiki_tmp_asqa-0.9.pkl'

ground_truth = np.load(gt_file)
gt_qi = np.load(gt_qi_file)
assert len(gt_qi) == ground_truth.shape[0], f"Expected {len(gt_qi)} == {ground_truth.shape[0]}"

with open(ret_pkl, 'rb') as f:
    retrieved, rdist = pickle.load(f)

recall = pysvs.k_recall_at(ground_truth, retrieved[gt_qi], 10, 10)
print(f"Recall = {recall}")

