#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, December 14th 2023, 2:01:36 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
#
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import random
import logging

class cn_college_listen_mcq_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']
            instruction = 'Question:\n' + sample['instruction'] + '\n Choices:\n' + sample['choices']
            reference   = sample['answer']
            input_data.append({
                                "audio"    : audio,
                                "text"     : instruction,
                                "answer"   : reference,
                                "task_type": "SQA",
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions):

        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions


    def compute_score(self, data_with_model_predictions, metrics=None):
        
        questions   = []
        references  = []
        predictions = []

        for item in data_with_model_predictions:
        
            question         = item["text"]
            answer           = item["answer"]
            model_prediction = item["model_prediction"]

            questions.append(question)
            references.append(answer)
            predictions.append(model_prediction)

        if metrics == 'llama3_70b_judge':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge
            llama3_70b_judge_results, all_details = llama3_70b_as_judge("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
            return {'llama3_70b_judge': llama3_70b_judge_results, 'details': all_details}

        elif metrics == 'llama3_70b_judge_binary':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
            llama3_70b_judge_binary_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
            return {'llama3_70b_judge_binary': llama3_70b_judge_binary_results, 'details': all_details}        
        
        elif metrics == 'llama3_8b_judge':
            from dataset_src.eval_methods.eval_llama3_8b import llama3_8b_as_judge
            llama3_8b_judge_results = llama3_8b_as_judge("../prepared_models/Meta-Llama-3-8B-Instruct-hf", [questions, references, predictions])
            return {'llama3_8b_judge': llama3_8b_judge_results}
        
        elif metrics == 'prometheus2_judge':
            from dataset_src.eval_methods.eval_prometheus2 import prometheus2_as_judge
            prometheus2_judge_results = prometheus2_as_judge("../prepared_models/prometheus-7b-v2.0", [questions, references, predictions])
            return {'prometheus2_judge': prometheus2_judge_results}
        
        elif metrics == 'gpt4_judge':
            from dataset_src.eval_methods.eval_gpt4 import gpt4_as_judge
            gpt4_judge_results = gpt4_as_judge("", [questions, references, predictions])
            return {'gpt4_judge': gpt4_judge_results}
        
        else:
            raise ValueError("Invalid metrics: {}".format(metrics))
