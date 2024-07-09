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


from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE



def prometheus2_as_judge(model_path, input_data):
    """ Compute the score of the model on the given data."""

    model = VLLM(model=model_path)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    # generation
    all_details = []
    questions, references, predictions = input_data

    for instruction, reference_answer, response in tqdm(zip(questions, references, predictions), total=len(questions)):


        rubric_data = {
            "criteria": "Does the model provide accurate, relevant, and contextually appropriate responses to user inquiries?",
            "score1_description": "The model frequently fails to understand or address the core of the user's inquiries, providing inaccurate, irrelevant, or inappropriate responses.",
            "score2_description": "The model occasionally recognizes the topic of inquiry but often provides responses that are not sufficiently accurate, detailed, or contextually relevant.",
            "score3_description": "The model usually understands the question and attempts to provide a relevant answer, yet the responses may sometimes lack detail, accuracy, or context.",
            "score4_description": "The model consistently understands and appropriately addresses the questions, providing accurate and relevant responses. However, there may still be minor inaccuracies or instances where additional context could enhance clarity.",
            "score5_description": "The model excels in understanding user inquiries and consistently delivers accurate, detailed, and contextually appropriate responses that thoroughly address the user's needs."
        }

        score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

        feedback, score = judge.single_absolute_grade(
            instruction=instruction,
            response=response,
            rubric=score_rubric,
            reference_answer=reference_answer
        )


        sample_rating_detail = {
            'question'        : instruction,
            'reference'       : reference_answer,
            'model_prediction': response,
            'judge_response'  : feedback,
            'rate_score'      : score,
            'success'         : 1,
        }

        all_details.append(sample_rating_detail)

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate, 'details': all_details}

    # Clear the model
    del model
    torch.cuda.empty_cache()

    return judge_results




