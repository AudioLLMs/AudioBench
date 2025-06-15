#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, June 7th 2024, 4:29:52 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import os
import torch
import transformers

from tqdm import tqdm

import random

from openai import OpenAI
from multiprocessing import Pool

import re
import json

def format_test(input, rule, target):
    if not rule:
        return True
    rule = str(rule)
    if rule == "1":
        return includes_word(input, target)
    elif rule == "2":
        return exclude_word(input, target)
    elif rule == "3":
        return all_letters_uppercase(input)
    elif rule == "4":
        return all_letters_lowercase(input)
    elif rule == "5":
        return is_word_uppercase(input, target)
    elif rule == "6":
        return is_word_lowercase(input, target)
    elif rule == "7":
        return starts_with(input, target)
    elif rule == "8":
        return ends_with(input, target)
    elif rule == "9":
        return is_quoted_by(input, target)
    elif rule == "10":
        return has_no_symbols(input)
    elif rule == "11":
        return is_valid_list_structure(input, target)
    elif rule == "12":
        return is_word_count_in_range(input, target)
    elif rule == "13":
        return is_valid_json_structure(input, target)
    else:
        print("ERROR - undefined rule!")
        return False

def includes_word(input, target):
    pattern = r'\b' + re.escape(target) + r'\b'
    return bool(re.search(pattern, input, re.IGNORECASE))

def exclude_word(input, target):
    pattern = r'\b' + re.escape(target) + r'\b'
    return not bool(re.search(pattern, input, re.IGNORECASE))

def all_letters_uppercase(input):
    letters = [char for char in input if char.isalpha()]
    return all(char.isupper() for char in letters) if letters else True

def all_letters_lowercase(input):
    letters = [char for char in input if char.isalpha()]
    return all(char.islower() for char in letters) if letters else True

def is_word_uppercase(input, target):
    pattern = r'\b' + re.escape(target) + r'\b'
    matches = re.findall(pattern, input, re.IGNORECASE)
    if not matches:
        return True
    return any(match.isupper() for match in matches)

def is_word_lowercase(input, target):
    pattern = r'\b' + re.escape(target) + r'\b'
    matches = re.findall(pattern, input, re.IGNORECASE)
    if not matches:
        return True
    return any(match.islower() for match in matches)

def starts_with(input, target):
    if not target:
        return False
    return input.startswith(target)

def ends_with(input, target):
    if not target:
        return False
    return input.endswith(target)

def is_quoted_by(input, target):
    if not target:
        return False

    if target == "()":
        return input.startswith("(") and input.endswith(")")
    elif target == "[]":
        return input.startswith("[") and input.endswith("]")
    elif target == "{}":
        return input.startswith("{") and input.endswith("}")
    elif target == "<>":
        return input.startswith("<") and input.endswith(">")
    else:
        return input.startswith(target) and input.endswith(target)

def has_no_symbols(input):
    return not bool(re.search(r'[^\w\s]|\_', input))

def is_valid_list_structure(input, target):
    if not target or target not in ["0", "1", "2", "3"]:
        return False
    
    lines = [line.strip() for line in input.splitlines() if line.strip()]
    if not lines:
        return True
    
    patterns = {
        "0": r'^-',
        "1": r'^\d+\.',
        "2": r'^[IVXLCDM]+\.',
        "3": r'^[A-Z]\.'
    }
    
    pattern = patterns[target]
    
    if not any(re.match(pattern, line) for line in lines):
        return False
    
    if target == "0":
        return True
    
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    def roman_to_int(s):
        result = 0
        prev_value = 0
        for char in reversed(s):
            curr_value = roman_values[char]
            if curr_value >= prev_value:
                result += curr_value
            else:
                result -= curr_value
            prev_value = curr_value
        return result
    
    markers = []
    for line in lines:
        if re.match(pattern, line):
            match = re.match(pattern, line)
            marker = match.group(0)
            if target == "1":
                num = int(marker[:-1])
                markers.append(num)
            elif target == "2":
                roman = marker[:-1]
                num = roman_to_int(roman)
                markers.append(num)
            elif target == "3":
                letter = marker[:-1]
                num = ord(letter) - ord('A') + 1
                markers.append(num)
    
    if not markers:
        return False
    for i in range(len(markers) - 1):
        if markers[i + 1] != markers[i] + 1:
            return False
    
    return True

def is_word_count_in_range(input, limit):
    try:
        bounds = limit.split("-")
        if len(bounds) != 2:
            print("ERROR - insufficient bounds!")
            return False
        lower_limit = int(bounds[0])
        upper_limit = int(bounds[1]) if int(bounds[1]) != 0 else 99999
    except (ValueError, IndexError):
        return False
    
    word_count = len(input.split())
    return lower_limit <= word_count <= upper_limit

def is_valid_json_structure(input, target):
    def normalize_and_unescape(s):
        # Extract content between first { and last } (inclusive)
        first_brace = s.find('{')
        last_brace = s.rfind('}')
        if first_brace == -1 or last_brace == -1 or first_brace > last_brace:
            return None
        s = s[first_brace:last_brace + 1]

        s = s.strip()
        s = re.sub(r'\\{2,}', r'\\', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s = s.replace('\\"', '"')
            s = re.sub(r'(\w+):', r'"\1":', s)
            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                return None

    input_json = normalize_and_unescape(input)
    # print(f"Input JSON: {input_json}")
    if input_json is None:
        return False
    
    target_json = normalize_and_unescape(target)
    if target_json is None:
        # print(f"Target JSON parsing failed")
        return False
    
    if not (isinstance(input_json, dict) and isinstance(target_json, dict)):
        # print("Not both dictionaries")
        return False
    
    def check_structure(input_obj, target_obj):
        # print(f"Checking structure: {input_obj} against {target_obj}")
        if isinstance(target_obj, dict):
            if not isinstance(input_obj, dict):
                # print("Input is not a dict")
                return False
            input_keys = [key.strip() if isinstance(key, str) else key for key in input_obj.keys()]
            target_keys = [key.strip() if isinstance(key, str) else key for key in target_obj.keys()]
            # print(f"Input keys: {input_keys}, Target keys: {target_keys}")
            if set(input_keys) != set(target_keys):
                # print("Keys do not match")
                return False
            return all(check_structure(input_obj[key], target_obj[key]) for key in target_obj)
        
        if isinstance(target_obj, list):
            # print(f"Checking list: {isinstance(input_obj, list)}")
            return isinstance(input_obj, list)
        
        input_type = type(input_obj)
        target_type = type(target_obj)
        # print(f"Types: input {input_type}, target {target_type}")
        if target_type == str:
            return input_type == str
        if target_type == int or target_type == float:
            return input_type == int or input_type == float
        if target_type == bool:
            return input_type == bool
        if target_obj is None:
            return input_obj is None
        return False
    
    result = check_structure(input_json, target_json)
    # print(f"Final result: {result}")
    return result

def llama3_70b_as_judge_binary_one_sample(args):

    tokenizer, question, reference, prediction, dimension, rule, rule_target = args

    PROMPT_TEMPLATE = """\
        [Reference Answer]
        {reference}

        [Model Answer]
        {prediction}

        [Question]
        {question}

        [Task]
        Rate the model's answer based on its alignment with the reference answer, focusing on the following two aspects:
        1. **Correctness**: Assess if the model's answer demonstrates the correct understanding and response based on the [Reference Answer].
            Score 0: If the question is regarding to transcriptions, the model's answer is not the same as [Reference Answer]. If the question is not regarding to transcriptions, the model's answer does not accurately reflect the meaning or idea of [Reference Answer].
            Score 1: If the question is regarding to transcriptions, the model's answer is exactly the same as [Reference Answer]. If the question is not regarding to transcriptions, the model's answer accurately reflects the meaning or idea of [Reference Answer].
            
        Please provide two separate ratings:
        1. **Correctness Rating**: (0 or 1)

        Your final output should be exactly in this format, you can only modify contents inside brackets.
        Correctness Rating: (int)
        Explanation: (Provide a concise explanation for each rating. For **Correctness**, explain if the model’s answer is correct and aligns with the reference. For **Instruction-Following**, explain how well the model adhered to the task instructions and any discrepancies.)
        """

    evaluation_prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, reference=reference)

    messages = [
        {"role": "user", "content": evaluation_prompt},
    ]

    templated_sample = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False,
    )

    # Model
    port = os.environ.get('MY_VLLM_PORT_JUDGE', 5001)
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    completion = client.completions.create(
    model      = model,
    prompt     = templated_sample,
    max_tokens = 512,
    n          = 1,
        )
    
    output = completion.choices[0].text.strip()

    try:
        ratings = output.split("\n")
        
        correctness_rating = int(ratings[0].split(":")[-1].strip())  # Extract Correctness Rating
        instruction_following_rating = 1 if format_test(prediction, rule, rule_target) else 0

        success = 1 if correctness_rating == 1 and instruction_following_rating == 1 else 0
    except:
        correctness_rating = 0
        instruction_following_rating = 0
        success = 0

    sample_rating_detail = {
        'question': question,
        'reference': reference,
        'model_prediction': prediction,
        'judge_response': output,
        'correctness_rating': correctness_rating,
        'instruction_following_rating': instruction_following_rating,
        'success': success,
        'dimension': dimension
    }

    return sample_rating_detail



def llama3_70b_as_judge_binary(model_path, input_data):
    """ Compute the score of the model on the given data."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Generation
    questions, references, predictions, dimensions, rules, rule_targets = input_data

    num_processes = min(8, len(input_data))
    

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(llama3_70b_as_judge_binary_one_sample, zip([tokenizer]*len(questions), questions, references, predictions, dimensions, rules, rule_targets)),
                total=len(input_data),
                desc="Processing"
            )
        )

    correctness_ratings = [detail['correctness_rating'] for detail in all_details]
    avg_correctness_ratings = sum(correctness_ratings) / len(correctness_ratings) * 100
    instruction_following_ratings = [detail['instruction_following_rating'] for detail in all_details]
    avg_instruction_following_ratings = sum(instruction_following_ratings) / len(instruction_following_ratings) * 100
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    dimension_success_rates = {
        'Content Requirements': {'correctness': 0, 'instruction_following': 0, 'success': 0, 'count': 0},
        'Capitalization Requirements': {'correctness': 0, 'instruction_following': 0, 'success': 0, 'count': 0},
        'Symbol Rules': {'correctness': 0, 'instruction_following': 0, 'success': 0, 'count': 0},
        'List and Structure Requirements': {'correctness': 0, 'instruction_following': 0, 'success': 0, 'count': 0},
        'Length Requirements': {'correctness': 0, 'instruction_following': 0, 'success': 0, 'count': 0},
        'Format Requirements': {'correctness': 0, 'instruction_following': 0, 'success': 0, 'count': 0},
    }

    for detail in all_details:
        dimension = detail.get('dimension')

        if dimension in dimension_success_rates:
            dimension_success_rates[dimension]['correctness'] += detail['correctness_rating']
            dimension_success_rates[dimension]['instruction_following'] += detail['instruction_following_rating']
            dimension_success_rates[dimension]['success'] += detail['success']
            dimension_success_rates[dimension]['count'] += 1

    for dimension, stats in dimension_success_rates.items():
        if stats['count'] > 0:
            stats['avg_correctness'] = stats['correctness'] / stats['count']
            stats['avg_instruction_following'] = stats['instruction_following'] / stats['count']
            stats['success_rate'] = stats['success'] / stats['count']
        else:
            stats['avg_correctness'] = 0
            stats['avg_instruction_following'] = 0
            stats['success_rate'] = 0

    judge_results = {'judge_correctness_rating': avg_correctness_ratings, 'judge_instruction_following_ratings': avg_instruction_following_ratings, 'success_rate': success_rate, 'dimensional_success_rate': dimension_success_rates}

    return judge_results, all_details






