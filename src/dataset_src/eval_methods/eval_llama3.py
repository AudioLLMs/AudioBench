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

import torch
import transformers

from tqdm import tqdm

import random

from openai import OpenAI


def llama3_as_judge(model_path, input_data):
    """ Compute the score of the model on the given data."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    #model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    #model.eval() # set to eval mode, by default it is in eval model but just in case

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # generation
    all_details = []
    questions, references, predictions = input_data

    for question, reference, prediction in tqdm(zip(questions, references, predictions), total=len(questions)):

        
        PROMPT_TEMPLATE = """\
            [Reference Answer]
            {reference}

            [Model Answer]
            {prediction}

            [Question]
            {question}

            [Task]
            Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
            Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
            Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
            Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
            Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
            Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
            Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
            Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

            Your response should be formatted as follows:
            Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
            Rating: (int)"""



        evaluation_prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, reference=reference)

        messages = [
            #{"role": "system", "content": "You are an expert grader!"},
            {"role": "user", "content": evaluation_prompt},
        ]

        templated_sample = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )


        
        # Model
        port = random.choice([5000])
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
            rate_score = float(output.split()[-1])
            success = 1
        except:
            rate_score = 0.0
            success = 0

        sample_rating_detail = {
            'question'        : question,
            'reference'       : reference,
            'model_prediction': prediction,
            'judge_response'  : output,
            'rate_score'      : rate_score,
            'success'         : success,
        }

        all_details.append(sample_rating_detail)

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate, 'details': all_details}

    return judge_results





