import argparse
import collections
from collections import Counter
import re
import torch
import copy
import os

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

import string
from evaluate import load

from transformers import logging as t_log
t_log.set_verbosity_error()  # supress bert warning

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

from utils import normalize_answer, get_max_memory, remove_citations, convert_textual_numbers_to_numeric

import logging

"""
Script for obtaining reader eval metrics for generated text, tailored to each dataset

ASQA
- string exact match (str_em), following ALCE
- string hit (str_hit), following ALCE
- rougeLsum, following ALCE

NQ,
- substring match, following RAGGED
- f1, following RAGGED
- rougel f1/precision/recall

"""


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#RESULTS_PATH = os.environ.get("RESULTS_PATH")
DATA_PATH = os.environ.get("DATA_PATH")
from file_utils import load_json, save_json


AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


def compute_f1(a_gold, a_pred):
    """
    Compute F1 score between two strings based on token overlap
    """

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_str_em(data):
    """
    For each generation, checks if the gold short answers are substrings of the generation (for it to be completely correct, the generation must include ALL)

    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM: report the mean that a gold string is in the corresponding generation over the number of gold strings
        STR-EM-HIT: reports the mean that ALL gold strings were in the generation over the number of generations
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["generated_output"]))
        acc.append(100 * np.mean(loc_acc))
        hit.append(100 * int(np.mean(loc_acc) == 1))

    return acc, hit


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_autoais(data,
                    qampari=False,
                    at_most_citations=None,
                    noise=False,
                    ndoc=None,
                    nrand=None,
                    noise_first=False,
                    noise_file=None,
                    gold_run=False):
    """
    Compute Auto-Attributable to Identified Sources (Auto-AIS) score.
    Automated metric for approximating whethern a snippet of text can be inferred from a larger text
    used here to approximate whether a sentence can be attributed to its corresponding cited source
    https://arxiv.org/pdf/2210.08726 

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
    Returns: 
        dict with keys
            citation_rec - 1 if the joint set of citations fully support the statement, averaged over all statements w/ 1+ citations
            citation_prec- 1 if the rec is 1 (for a statement the citation supports) AND if the citation fully or partially supports the statement, averaged over all citations
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])

    nq = len(data)
    # Initialize query-level output variables
    query_metadata = []
    joint_entailment = []
    citation_necessity = []
    all_refs = []
    num_sent = np.zeros((nq, ))
    unattributed_count = np.zeros((nq, ))
    citations_oor_count = np.zeros((nq, ))

    sent_mcite = 0  # Sentences with multiple citations
    sent_mcite_support = 0
    sent_mcite_overcite = 0

    if noise:
        noise_path = os.path.join(DATA_PATH, noise_file)
        noise_items = load_json(noise_path, logger)
    if gold_run:
        logger.info("Assuming all retrieved documents are gold")

    # Loop over query items
    for item_idx, item in enumerate(tqdm(data)):
        # Segment into sentences & remove citations
        if qampari:  # split by comma instead of tokenizing (strips off whitespace, periods and commas before split)
            sents = [item['question'] + " " + x.strip() for x in item['generated_output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['generated_output'])
        
        # if noisy experiment, reorder doc list
        if noise:
            noise_docs = (noise_items[item_idx]['docs'])[:nrand]
            retrieved_docs = item['docs'][:ndoc]
            if noise_first:
                doc_list = noise_docs + retrieved_docs
            else:
                doc_list = retrieved_docs + noise_docs
        else:
            doc_list = item['docs']  # Returns all retrieved documents, e.g. 100 documents
            # Figure out how many documents were actually given in the prompt
            ref_num = re.findall("\[(\d+)\]", item['prompt'])
            ref_num = np.array([int(n) for n in ref_num])
            max_in_prompt = ref_num.max()
            doc_list = doc_list[:max_in_prompt]

        # Get gold documents for this query item
        if gold_run:
            doc_is_gold = [True] * len(doc_list)
            ngold = len(doc_list)
        else:
            if 'id_set' in item['output']:
                gold_ids = item['output']['id_set']
                doc_is_gold = [str(doc['id']) in gold_ids for doc in doc_list]
            else:
                gold_ids = item['output']['page_par_id_set']
                doc_is_gold = [f"{doc['id']}_{doc['start_par_id']}" in gold_ids for doc in doc_list]
            ngold = len(gold_ids)

        # Save out key details here
        meta = {"question": item['question'],
                "generated_output": item['generated_output'],
                "sent_output": sents,
                "retrieved_docs": doc_list,
                "doc_is_gold": doc_is_gold,
                "ngold": ngold}
        query_metadata.append(meta)

        if len(sents) == 0:
            logger.info(f"No generation found for query {item_idx}, skipping AIS checks...")
            all_refs.append([])
            citation_necessity.append([])
            joint_entailment.append([])
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]  # strip citations
        # Initialize variables for this generated passage
        unattributed_sents = 0
        citations_oor = 0
        joint_entail = np.ones((len(sents), )) * -1
        entail_sents = []
        sent_refs = []
        for sent_id, sent in enumerate(sents):
            entail_per_sent = {}
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            sent_refs.append(ref)
            #logger.info(f"For `{sent}`, find citations {ref}")  # verified extracting refs
            if len(ref) == 0:
                # No citations
                joint_entail[sent_id] = 0
                unattributed_sents += 1
            elif any([ref_id >= len(doc_list) for ref_id in ref]):
                # Any citation out of range -- automatically gets joint entailment = 0
                joint_entail[sent_id] = 0
                citations_oor += 1

            # 9/6: Changing to evaluate joint entailment regardless of gold doc
            #if not any([doc_is_gold[ref_id] for ref_id in ref if ref_id < len(doc_list)]):
            #    # None of the citations has a gold document
            #    joint_entail[sent_id] = 0
            #    entail_sents.append(entail_per_sent)
            #    continue
            #else:
            if at_most_citations is not None:
                ref = ref[:at_most_citations]
            joint_passage = '\n'.join([_format_document(doc_list[psgs_id]) for psgs_id in ref if psgs_id < len(doc_list)])

            # If not directly rejected by citation format error and has gold doc, calculate the recall score
            if joint_entail[sent_id] == -1:
                joint_entail[sent_id] = _run_nli_autoais(joint_passage, target_sent)

            if len(ref) > 1:
                sent_mcite += 1  # multiple citations in this sentence
            # Determine necessity of each citation
            if joint_entail[sent_id] and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                i_supports, setminusi_supports = np.empty((len(ref),)), np.empty((len(ref),))
                i_supports[:], setminusi_supports[:] = np.nan, np.nan
                # Loop over citations that support this sentence
                for pii, psgs_id in enumerate(ref):
                    if not doc_is_gold[psgs_id]:
                        # Skip the NLI computation if not citing a gold document
                        continue
                    # condition A for irrelevance: statement not entailed by doc alone
                    passage = _format_document(doc_list[psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent)
                    i_supports[pii] = nli_result
                    # condition B for irrelevance: statement entailed by set without the doc
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(doc_list[pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        setminusi_supports[pii] = nli_result
                        if nli_result:  # psgs_id is not necessary
                            sent_mcite_overcite += 1
                entail_per_sent['citei_supports'] = i_supports
                entail_per_sent['setminusi_supports'] = setminusi_supports
            entail_sents.append(entail_per_sent)

        assert len(entail_sents) == len(sents), "Unexpected number of sentences in entail_sents"
        assert len(sent_refs) == len(sents), "Unexpected number of sentences in sent_efs"
        citation_necessity.append(entail_sents)
        joint_entailment.append(joint_entail)
        all_refs.append(sent_refs)

        num_sent[item_idx] = len(sents)
        unattributed_count[item_idx] = unattributed_sents
        citations_oor_count[item_idx] = citations_oor

    # After all queries have been processed, print some statistics
    assert len(citation_necessity) == nq, "Unexpected number of entail scores"
    assert len(joint_entailment) == nq, "Unexpected number of entail scores"
    assert len(all_refs) == nq, "Unexpected number of citations"
    assert len(query_metadata) == nq, "Unexpected number of query metadata entries"
    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / num_sent.sum(),
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    return {
        "num_sent": num_sent.tolist(),
        "unattributed_sents": unattributed_count,
        "citations_oor": citations_oor_count,
        "citation_necessity": citation_necessity,
        "joint_entailment": joint_entailment,
        "query_metadata": query_metadata,
        "citations_per_sent": all_refs
    }


def compute_ragged_metrics(normalized_data, NO_BERT, MERGE_LIST):
    """
    Original eval for NQ and BIOASQ from RAGGED paper
    """

    def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        """
        utility to get max over ground truth values
        """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)

        if not isinstance(scores_for_ground_truths[0], dict):
            return max(scores_for_ground_truths)
        elif 'rougeLsum_f1' in scores_for_ground_truths[0].keys():
            # get max entry by f1 score
            return max(scores_for_ground_truths, key=lambda x:x['rougeLsum_f1'])
        else:
            raise ValueError("score should be a dict or 'rougeLsum_f1'")

    def _rougel_score(prediction, ground_truth):
        # no normalization
        prediction = convert_textual_numbers_to_numeric(prediction)
        ground_truth = convert_textual_numbers_to_numeric(ground_truth)
        try:
            scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
            scores = scorer.score(prediction, ground_truth)
            scores = scores["rougeLsum"]
        except ValueError:  # "Hypothesis is empty."
            return 0.0

        return  {
            "rougeLsum_p": scores[0],
            "rougeLsum_r": scores[1],
            "rougeLsum_f1": scores[2]
        }
        

    def kilt_eval(guess_answer, gold_candidate_answers):

        # returns True if ANY of the gold candidates are in generated answer 
        substring_match = exact_presence(gold_candidate_answers, guess_answer)

        # returns max f1 over all gold answers
        # f1 is calculated as token overlap between gold and generated answer
        local_f1 = _metric_max_over_ground_truths(
            compute_f1, guess_answer, gold_candidate_answers
        )

        # max rougeLsum f1 over ground truths
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )
        
        return substring_match, local_f1, local_rougel

    total_count = 0

    all_norm_substr = []
    all_norm_f1 = []
    all_rougel_f1 = []
    all_rougel_p = []
    all_rougel_r = []

    for reader_output_info in tqdm(normalized_data):
        total_count+=1

        guess_answer = reader_output_info["generated_output"]
        gold_data = reader_output_info["output"]
        gold_answer_list = [x for x in gold_data["answer_set"]]  # considering only the short answers

        # merge gold answer list if bioasq specifies answer_type==list
        if MERGE_LIST and "question_type" in gold_data.keys() and gold_data["question_type"] == "list":
            gold_answer_list = [" ".join(gold_answer_list)]

        substring_match, local_f1, \
            local_rougel = kilt_eval(
                  guess_answer, 
                  gold_answer_list
               )

        all_norm_substr.append(int(substring_match))
        all_norm_f1.append(local_f1)
        all_rougel_f1.append(local_rougel["rougeLsum_f1"])
        all_rougel_p.append(local_rougel["rougeLsum_p"])
        all_rougel_r.append(local_rougel["rougeLsum_r"])
        
    method_metrics = {'ragged_substring_match': all_norm_substr,
                      'ragged_f1': all_norm_f1,
                      'ragged_rougel_f1': all_rougel_f1,
                      'ragged_rougel_p': all_rougel_p,
                      'ragged_rougel_r': all_rougel_r}

    logger.info(f"total questions - dev: {total_count}/{len(gold_data)}")
    logger.info("Reader metrics : ", method_metrics)
    
    return method_metrics


def main(args):
    #file_path = os.path.join(RESULTS_PATH, 'reader', args.f)
    file_path = args.f  # USES ENTIRE DATA PATH NOW
    data_with_config = load_json(file_path, logger)
    data = data_with_config['data'] # remove config

    # Truncate by newline and remove on the fly search result
    logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning("We replace any on the fly search result to standard bracket citation format.")
    for i in range(len(data)):
        # if prompt text is in output, remove
        if data[i]['prompt'] in data[i]['generated_output']:
            prompt_len = len(data[i]['prompt'])
            data[i]['generated_output'] = data[i]['generated_output'][(prompt_len+1):]

        data[i]['generated_output'] = data[i]['generated_output'].strip().split("\n")[0]
        data[i]['generated_output'] = data[i]['generated_output'].replace("<|im_end|>", "")

    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['generated_output'] = remove_citations(normalized_data[i]['generated_output'])

    result = {}
    if "asqa" in args.f:
        result['str_em'], result['str_hit'] = compute_str_em(normalized_data)
    elif 'qampari' in args.f:
        raise NotImplementedError("No current support for QAMPARI + AutoAIS per query")
        #result.update(compute_qampari_f1(normalized_data, cot=args.cot))
        #qampari = True
    elif 'nq' in args.f:
        result.update(compute_ragged_metrics(normalized_data, NO_BERT=True, MERGE_LIST=False))
    elif 'bioasq' in args.f:
        raise NotImplementedError("No current support for BioASQ + AutoAIS per query")

    result.update(compute_autoais(
        data,
        qampari=False,
        at_most_citations=None,
        noise=args.noise, ndoc=args.ndoc, nrand=args.nrand,
        noise_first=args.noise_first, noise_file=args.noise_file,
        gold_run='gold' in file_path
    ))

    output_file = file_path + "_ais.score"
    import pandas as pd
    df = pd.DataFrame(result)
    df.to_json(output_file, orient='records', lines=True)
    logger.info(f"Saved to {output_file}")
    #save_json(result, output_file, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Name of reader output file to evaluate in $RESULTS_PATH/reader. Should have field `question`, `generated_output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")

    # if the results were generated with random noise, declare the order of prompt and number of noisy documents
    parser.add_argument("--noise", action="store_true", help="Whether this eval is part of a noise experiment")
    parser.add_argument("--ndoc", type=int, default=10, help="Number of retrieved documents used in prompt")
    parser.add_argument("--nrand", type=int, default=10, help="Number of random noise documents added to prompt")
    parser.add_argument("--noise_first", action="store_true", help="In the prompt, if random noisy documents should precede retrieved or gold passages")
    parser.add_argument("--noise_file", type=str, default=None, help="File from which noisy documents were added")

    args = parser.parse_args()
    main(args)
