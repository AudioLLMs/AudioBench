import random
import logging

import re

from jiwer import compute_measures, wer

from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr
# from dataset_src.prompts.prompts import asr_instructions

from pythainlp.tokenize import word_tokenize


class gigaspeech2_viet_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        # self.prompt   = asr_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']
            instruction = sample['instruction']
            reference   = sample['answer']
            input_data.append({
                                "audio"      : audio,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task_type"  : "ASR"
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

        if metrics != 'wer':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'wer' for ASR")

        def preprocess_vietnamese(text):
            """ Normalize Vietnamese text (lowercase, remove punctuation). """
            text = text.lower().strip()  # Convert to lowercase
            text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)  # Remove punctuation
            return text

        def merge_spaces(text):
            return re.sub(r'\s+', ' ', text).strip()

        predictions=[]
        references=[]
        for item in data_with_model_predictions:
            model_prediction = preprocess_vietnamese(item["model_prediction"])
            answer           = preprocess_vietnamese(item["reference"])

            model_prediction = merge_spaces(model_prediction)
            answer           = merge_spaces(answer)
            
            if len(model_prediction) == 0: model_prediction = "empty"
            if len(answer) == 0: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        sample_wer = []
        incorrect  = 0
        total      = 0
        for prediction, reference in zip(predictions, references):
            measures   = compute_measures(reference, prediction)
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total     += measures["substitutions"] + measures["deletions"] + measures["hits"]

            wer_score = wer(reference, prediction)

            sample_wer_score = {
                "reference" : reference,
                "prediction": prediction,
                "wer"       : wer_score,
            }

            sample_wer.append(sample_wer_score)

        total_wer = incorrect / total

        return {"wer": total_wer, "sample_wer": sample_wer}