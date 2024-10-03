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

QAMPARI
- precision, following ALCE
- recall, following ALCE
- recall top 5, following ALCE
- f1, following ALCE
- f1 top 5, following ALCE

NQ
- substring match, following RAGGED
- f1, following RAGGED
- rougel f1/precision/recall
"""


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESULTS_PATH = os.environ.get("RESULTS_PATH")
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


def compute_rouge(data):
    """
    Main function for rouge scoring: "automatically determines the quality of a summary by comparing it to other (ideal) summaries created by humans"

    Returns rougeLsum: splits the text into sentences based on newlines and computes the longest common subsequence between gold and generation for each 
    pair of sentences, then takes the union of all LCS scores
    https://aclanthology.org/W04-1013/

    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `generated_output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(
        hypotheses,
        references1,
        references2=[],
        metrics=['rougeLsum']
    ):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["generated_output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


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
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), np.std(acc), 100 * np.mean(hit), np.std(hit)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item['generated_output'].split())
        cntr += 1
    return res / cntr


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
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None,
                    noise=False,
                    ndoc=None,
                    nrand=None,
                    noise_first=False,
                    noise_file=None ):
    """
    Compute Auto-Attributable to Identified Sources (Auto-AIS) score.
    Automated metric for approximating whethern a snippet of text can be inferred from a larger text
    used here to approximate whether a sentence can be attributed to its corresponding cited source
    https://arxiv.org/pdf/2210.08726 

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
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


    ais_scores = []
    ais_scores_prec = []
    num_sent = []
    unattributed_count = []
    citations_oor_count = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []

    if noise:
        noise_path = os.path.join(DATA_PATH, noise_file)
        noise_items = load_json(noise_path, logger)

    for item_idx, item in enumerate(tqdm(data)):
        # Get sentences by using NLTK
        if qampari:  # split by comma instead of tokenizing (strips off whitespace, periods and commas before split)
            sents = [item['question'] + " " + x.strip() for x in item['generated_output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['generated_output'])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0

        unattributed_sents = 0
        citations_oor = 0

        # if noisy experiment, reorder doc list
        if noise:
            noise_docs = (noise_items[item_idx]['docs'])[:nrand]
            retrieved_docs = item['docs'][:ndoc]
            if noise_first:
                doc_list = noise_docs + retrieved_docs
            else:
                doc_list = retrieved_docs + noise_docs

        else:
            doc_list = item['docs']

        for sent_id, sent in enumerate(sents):

            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")  # verified extracting refs
            if len(ref) == 0:
                # No citations
                joint_entail = 0
                unattributed_sents += 1
            elif any([ref_id >= len(doc_list) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
                citations_oor += 1
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(doc_list[psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['generated_output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail

            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(doc_list[psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(doc_list[pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1 
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 

        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

        num_sent.append(len(sents))
        unattributed_count.append(unattributed_sents)
        citations_oor_count.append(citations_oor)
        

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total, 
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    ais_scores = [100 * score for score in ais_scores]
    ais_scores_prec = [100 * score for score in ais_scores_prec]

    return {
        "citation_rec_mean": np.mean(ais_scores),
        "citation_rec_std": np.std(ais_scores),
        "citation_prec_mean": np.mean(ais_scores_prec),
        "citation_prec_std": np.std(ais_scores_prec),
        "num_sent_mean": np.mean(num_sent),
        "unattributed_sents_mean": np.mean(unattributed_count),
        "citations_oor": np.mean(citations_oor_count)
    }


def compute_qampari_f1(data, cot=False):
    """
    Compute qampari-specific f1: splits generation by comma and calculates precision and recall based on this and list of gold entities,
    returns average over inputs

     Args:
        data: requires field `generated_output` and `answers`
              - answers: comma separated list of entities 
        cot: whether answers were generated with cot prompting
    """

    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item['generated_output']:
                o = ':'.join(item['generated_output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['generated_output']
        
        # remove leading/trailing space, period or comma -> split by comma and normalize
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        num_preds.append(len(preds))

        # clean and flatten answers
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]
        
        # 1 if prediction in gold, divide by number of predictions
        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
        # 1 if any of the predictions are in gold, divide by number of answers
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        # 1 if any of the predictions are in the top 5 answers, divide by 5
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))

        # calculate f1 over list of precision and recall
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0) 
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    prec = [100 * p for p in prec]
    rec = [100 * r for r in rec]
    rec_top5 = [100 * r for r in rec_top5]
    f1 = [100 * f for f in f1]
    f1_top5 = [100 * f for f in f1_top5]

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec_mean": np.mean(prec),
        "qampari_prec_std": np.std(prec),
        "qampari_rec_mean": np.mean(rec),
        "qampari_rec_std": np.std(rec),
        "qampari_rec_top5_mean": np.mean(rec_top5),
        "qampari_rec_top5_std": np.std(rec_top5),
        "qampari_f1_mean": np.mean(f1),
        "qampari_f1_std": np.std(f1),
        "qampari_f1_top5_mean": np.mean(f1_top5),
        "qampari_f1_top5_std": np.std(f1_top5)
    }

def compute_ragged_metrics(normalized_data, MERGE_LIST):
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
            raise ValueError("Cannot take max over the given scores")

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
    normalized_substring_match = 0
    normalized_f1 = 0
    rougel_f1 = 0
    rougel_p = 0
    rougel_r = 0
    
    logger.info("Running kilt evaluation...")

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
                  gold_answer_list,
               )

        normalized_substring_match += substring_match
        normalized_f1 += local_f1
        rougel_f1 += local_rougel["rougeLsum_f1"]
        rougel_p += local_rougel["rougeLsum_p"]
        rougel_r += local_rougel["rougeLsum_r"]

    if total_count > 0:
        normalized_substring_match /= total_count
        normalized_f1 /= total_count
        rougel_f1 /= total_count
        rougel_p /= total_count
        rougel_r /= total_count

    method_metrics = {
        "ragged_substring_match":round(normalized_substring_match, 4),
        "ragged_f1": round(normalized_f1, 4),
        "ragged_rougel_f1": round(rougel_f1, 4),
        "ragged_rougel_p": round(rougel_p, 4),
        "ragged_rougel_r": round(rougel_r, 4)

    }

    logger.info(f"total questions - dev: {total_count}/{len(gold_data)}")
    logger.info("Reader metrics : ", method_metrics)
    
    return method_metrics


def main(args):
    
    file_path = os.path.join(RESULTS_PATH, 'reader', args.f)
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
    result['length'] = compute_len(normalized_data)

    qampari = False  # used for different preprocessing during citation eval
    if "asqa" in args.f:
        result['str_em_mean'], result['str_em_std'], \
            result['str_hit_mean'], result['str_hit_std'] = compute_str_em(normalized_data)
        result['rougeLsum'] = compute_rouge(normalized_data)

    elif 'qampari' in args.f:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))
        qampari = True
    
    elif 'nq' in args.f or 'bioasq' in args.f:
        result.update(compute_ragged_metrics(normalized_data, args.merge_list_answers))
    
    if args.citations: 
        result.update(compute_autoais(
            data, qampari,
            at_most_citations=args.at_most_citations,
            noise=args.noise, ndoc=args.ndoc, nrand=args.nrand,
            noise_first=args.noise_first, noise_file=args.noise_file
        ))

    logger.info(result)
    output_file = file_path + ".score"
    save_json(result, output_file, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Name of reader output file to evaluate in $RESULTS_PATH/reader. Should have field `question`, `generated_output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    

    parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")

    # if the results were generated with random noise, declare the order of prompt and number of noisy documents
    parser.add_argument("--noise", action="store_true", help="Whether this eval is part of a noise experiment")
    parser.add_argument("--ndoc", type=int, default=10, help="Number of retrieved documents used in prompt")
    parser.add_argument("--nrand", type=int, default=10, help="Number of random noise documents added to prompt")
    parser.add_argument("--noise_first", action="store_true", help="In the prompt, if random noisy documents should precede retrieved or gold passages")
    parser.add_argument("--noise_file", type=str, default=None, help="File from which noisy documents were added")



    # QAMPARI
    parser.add_argument("--cot", action="store_true", help="For QAMPARI, try to find colon and separate the COT and answer listing")

    args = parser.parse_args()
    main(args)
