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

ds_instructions = [
    "Can you assist with summarizing the details from the audio?",
    "Would you mind helping to summarize the content of the audio?",
    "Can you break down the main ideas from the audio for me?",
    "Is it possible for you to summarize what's covered in the audio?",
    "Could you help provide a summary of the audio details?",
    "Can you summarize the key points from the audio?",
    "Could you help by giving an overview of the audio's content?",
    "Would you summarize the information from the audio for me?",
    "Can you assist in summarizing the audio recording?",
    "Could you break down the important parts of the audio for me?",
    "Is it possible for you to offer a summary of the audio?",
    "Can I get your help to summarize what’s in the audio?",
    "Could you outline the main points from the audio for me?",
    "Can you help clarify the main ideas in the audio?",
    "Would you be able to summarize the content captured in the audio?",
    "Could you offer a brief summary of what the audio contains?",
    "Can you assist me in summarizing the audio file’s content?",
    "Would you help me summarize the core message of the audio?",
    "Could you provide a brief overview of the audio’s information?",
    "Is it possible for you to break down the key ideas from the audio?",
    "Please summarize the audio content.",
    "Could you please summarize the audio?",
    "Please provide a summary of the audio.",
    "I would appreciate it if you could summarize the audio content.",
    "Could you kindly summarize the audio's content?"
    ]


class imda_part6_30s_ds_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.prompt   = ds_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']['audio']
            instruction = random.choice(self.prompt)
            reference   = sample['answer']['text']
            input_data.append({
                                "audio"    : audio,
                                "text"     : instruction,
                                "answer"   : reference,
                                "task_type": "DS"
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
        
        elif metrics == 'gpt4o_judge':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge
            gpt4o_judge_results, all_details = gpt4o_as_judge("", [questions, references, predictions])
            return {'gpt4o_judge': gpt4o_judge_results, 'details': all_details}
        
        elif metrics == 'gpt4o_judge_binary':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
            gpt4o_judge_binary_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
            return {'gpt4o_judge_binary': gpt4o_judge_binary_results, 'details': all_details}
        
        else:
            raise ValueError("Invalid metrics: {}".format(metrics))


