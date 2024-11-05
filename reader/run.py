# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Some of this code is based on prior work under the MIT License:
#   Copyright (c) 2023 Princeton Natural Language Processing

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import re
import yaml
from reader.utils import *
from nltk import sent_tokenize

from file_utils import load_json, save_json

DATA_PATH = os.environ.get("DATA_PATH")
RESULTS_PATH = os.environ.get("RESULTS_PATH")

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

class LLM:

    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.model, self.tokenizer = load_model(self.model_name)
        
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0

    def tokenize(self, prompt):
        if 'mistral' in self.model_name.lower():
            return mistral_tokenizer(self.tokenizer, prompt, convert_to_torch=False)
        else: 
            return self.tokenizer.tokenize(prompt)


    def generate(self, prompt, max_tokens, stop=None):

        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")
            
        if 'mistral' in self.model_name.lower():
            prompt_len = len(prompt)
            inputs = mistral_tokenizer(self.tokenizer, [prompt], convert_to_torch=True).to(self.model.device)  # or [prompt]??
            outputs = self.model.generate(
                inputs,  #?
                attention_mask=torch.ones_like(inputs),
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
                eos_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            generation = self.tokenizer.decode(outputs[0].tolist())
            generation = generation[(prompt_len+1):] if len(generation) > prompt_len else generation

        else: 
            logger.info("generating...")
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

            stop = [] if stop is None else stop
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [self.model.config.eos_token_id]))
            if "llama" in args.model.lower():
                # logger.info("removing unk ids")
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            outputs = self.model.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=stop_token_ids
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        return generation


def main(args):
    
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    prompt_path = os.path.join("reader", "prompts", args.prompt_file)
    prompt_data = load_json(prompt_path, logger)

    eval_path = os.path.join(DATA_PATH, args.eval_file)
    eval_data = load_json(eval_path, logger)

    noise_data = None
    if args.noise_file: 
        noise_file  = os.path.join(DATA_PATH, args.noise_file)
        noise_data = load_json(noise_file, logger)


    # Generate the demonstration part
    head_prompt = ""
    if args.shot > 0:
        train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
        for train_id in train_ids:
            train_item = prompt_data["demos"][train_id]
            ndoc = args.ndoc
            if args.no_doc_in_demo:
                ndoc = 0
            elif args.fewer_doc_in_demo:
                assert args.ndoc_in_demo is not None
                ndoc = args.ndoc_in_demo
            head_prompt += make_demo(
                train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
                instruction=prompt_data["instruction"], use_shorter=args.use_shorter 
            )
            head_prompt += prompt_data["demo_sep"]

    # Sample quick test
    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):
        noise_entry = None
        if noise_data:
            noise_entry = noise_data[idx]
            # logger.info(noise_entry.keys())
        eval_data[idx]['prompt'] = head_prompt + make_demo(
            eval_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter, 
            test=True, noise_data=noise_entry, noise_first=args.noise_first, nrand=args.nrand
        )
        doc_list = get_shorter_text(eval_item, eval_item["docs"], args.ndoc, args.use_shorter) if args.use_shorter is not None else eval_item["docs"][:args.ndoc]

        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1

    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
            item['prompt'] = prompt
            
            output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            if output_array[-1].endswith("End."):
                output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Prompt length={prompt_len}")
            logger.info(f"Question: {item['question']}")
            logger.info(f"Gold answer: {item['answer']}")
            logger.info(f"Final model output: {output_array[-1]}") 
        
        item['generated_output'] = output_array if len(output_array) > 1 else output_array[0]
        
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.tag}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}"

    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"
    if args.no_doc_in_demo:
        name += "-no_doc_in_demo"
    if args.fewer_doc_in_demo:
        name += f"-{args.ndoc_in_demo}_doc_in_demo"
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
    if 'cite' in args.prompt_file:
        name += "-cite"
    

    if 'bge-base' in args.eval_file:
        name += '-bge-base'
    elif 'bge-large' in args.eval_file:
        name += '-bge-large'
    elif 'colbert' in args.eval_file:
        name += '-colbert'
    elif 'gold' in args.eval_file:
        name += '-gold'
    else:
        name += '-no-context'

    if args.noise_file and args.noise_first:
        name += '-noise-first'

    if args.add_name:
        if not (args.add_name).startswith("-"):
            args.add_name = "-" + args.add_name
        name += args.add_name
    
    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }

    file_path = os.path.join(RESULTS_PATH, "reader")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, name + ".json")
    save_json(eval_data, file_path, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Name of config file in reader/configs directory")

    # Prompt file is a json file that contains the following fields:
    # - instruction: the instruction, which will appear at the beginning of each demo and the test example
    # - demo_sep: the separator between each demo, for example, "\n\n\n"
    # - demo_prompt: the prompt for the demo, for example, "Instruction: {INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
    #     - {INST}: the instruction
    #     - {D}: the documents
    #     - {Q}: the question
    #     - {A}: the answers
    # - doc_prompt, the prompt for each document, for example, "Document [{ID}](Title: {T}): {P}", where
    #     - {ID}: the document id, staring from 1
    #     - {T}: the document title
    #     - {P}: the document text
    # - demos: a list of demo examples, each of which should have
    #     - question: the question
    #     - docs: the documents ("title" and "text")
    #     - answer: the answer to show in the demo. If it is a list, they will be concatenated by "\n". This is useful when the answer includes interactive components.
    # Note that this python file will sample `--shot` demos from the prompt file given the random seed `--seed`
    parser.add_argument("--prompt_file", type=str, help="name of prompt file in reader/prompts")

    # Evaluation file is a json file that contains a list of item, each of which contains
    # - question: the question
    # - answer: the answer
    # - docs: the documents, each of which contains "title", "text"
    parser.add_argument("--eval_file", type=str, help="name of eval file with neighbors")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents used for context")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="remove documents in demonstration examples, only showing example questions and answers in the demos")
    parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos, must use with --ndoc_in_demo to specify number of docs")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")

    parser.add_argument("--noise_file", type=str, default=None, help="Noise file in $DATA_DIR from which to add noise")
    parser.add_argument("--nrand", type=int, default=0, help="Number of random noise documents to add to prompt")
    parser.add_argument("--noise_first", action="store_true", help="In the prompt, if random noisy documents should precede retrieved or gold passages")
    parser.add_argument("--add_name", type=str, default=None, help="String to add to end of output file name")

    # Load config
    args = parser.parse_args()
    config_file = os.path.join("reader", "configs", args.config)
    config = yaml.safe_load(open(config_file)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    main(args)
