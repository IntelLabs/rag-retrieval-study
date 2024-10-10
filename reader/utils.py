import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time



def is_potential_number(word):
    """
    Check if a word is a potential part of a number in textual form.
    """
    number_parts = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
                    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", 
                    "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion"]
    return word.lower() in number_parts


def convert_textual_numbers_to_numeric(sentence):
    """
    Convert textual numbers within a sentence to numeric form, handling compound numbers.
    Args:
    - sentence (str): The sentence to process.
    
    Returns:
    - str: The sentence with numbers converted to numeric form.
    """
    from word2number import w2n
    words = sentence.split()
    converted_words = []
    current_number_phrase = []

    for word in words:
        if is_potential_number(word):
            current_number_phrase.append(word)
        else:
            if current_number_phrase:
                # Convert the current number phrase to a number
                number_string = " ".join(current_number_phrase)
                try:
                    numeric_value = w2n.word_to_num(number_string)
                    converted_words.append(str(numeric_value))
                except ValueError:
                    # If conversion fails, keep the original phrase
                    converted_words.extend(current_number_phrase)
                current_number_phrase = []
            
            converted_words.append(word)

    # Handle any remaining number phrase at the end
    if current_number_phrase:
        try:
            number_string = " ".join(current_number_phrase)
            numeric_value = w2n.word_to_num(number_string)
            converted_words.append(str(numeric_value))
        except ValueError:
            converted_words.extend(current_number_phrase)

    return ' '.join(converted_words)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = convert_textual_numbers_to_numeric(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-4}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    title = "" if not doc["title"] else doc["title"]
    return doc_prompt.replace("{T}", title).replace("{P}", text).replace("{ID}", str(doc_id+1))


def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False, noise_data=None, noise_first=False, nrand=10):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            if noise_data and noise_first:
                text = "".join([make_doc_prompt(doc, doc_id + nrand, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            else: 
                text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            if noise_data:
                rand_doc_list = get_shorter_text(noise_data, noise_data["docs"], nrand, use_shorter) if use_shorter is not None else noise_data["docs"][:nrand]
                if noise_first:
                    rand_text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(rand_doc_list)])
                    text = rand_text + "\n" + text
                else:
                    rand_text = "".join([make_doc_prompt(doc, doc_id + ndoc, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(rand_doc_list)])
                    text = text + "\n" + rand_text
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt

def mistral_tokenizer(tokenizer, text, convert_to_torch=False):
    import torch
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    if isinstance(text, list):
        token_list = [mistral_tokenizer(tokenizer, t, convert_to_torch) for t in text]
        if convert_to_torch:
            tokens = torch.stack(token_list, dim=0)
            assert tokens.ndim == 2, f"Expected tokens to be batch x seq"
            return tokens
        else:
            return token_list
    else:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=text)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        if convert_to_torch:
            tokens = torch.Tensor(tokens).to(torch.long)
        return tokens

def load_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model with hf library
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=dtype,
        max_memory=get_max_memory(),
        load_in_8bit=int8,
    )
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))
    
    if 'mistral'in model_name_or_path.lower():  # use mistral tokenizer
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        tokenizer = MistralTokenizer.v3()
    else: 
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        # Fix OPT bos token problem in HF
        if "opt" in model_name_or_path:
            tokenizer.bos_token = "<s>"
        tokenizer.padding_side = "left"

    return model, tokenizer
