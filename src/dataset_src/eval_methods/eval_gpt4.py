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

import openai
from openai import AzureOpenAI


def gpt4_as_judge(model_path, input_data):
    """ Compute the score of the model on the given data."""

    client = AzureOpenAI(
        azure_endpoint = 'https://janchatgpt.openai.azure.com', 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
        )

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
            {"role": "user", "content": evaluation_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model="gpt4-turbo-0409",
                messages = messages,
                temperature=0.7,
                max_tokens=500,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                )
            
            full_response = completion.choices[0].message.content
        except:
            print("Error in completion")
            full_response = "empty"
        
        # Map to scores
        try:
            rate_score = float(full_response.split()[-1])
            success = 1
        except:
            rate_score = 0.0
            success = 0

        sample_rating_detail = {
            'question'        : question,
            'reference'       : reference,
            'model_prediction': prediction,
            'judge_response'  : full_response,
            'rate_score'      : rate_score,
            'success'         : success,
        }

        all_details.append(sample_rating_detail)

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate, 'details': all_details}


    return judge_results







