import random
import logging

from dataset_src.math_utils import utils

def get_seperation_trigger(dataset: str):
    triggers = ['The answer is:', 'The answer is', 'the answer is']
    if dataset == 'gsm8k':
        triggers.append('####')
    return triggers


class spokenmqa_dataset_arithmatic(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        # self.prompt   = math_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']
            audio_gt    = sample['context_transcript']
            reference   = sample['answer']['text']
            instruction = sample['instruction']['text']
            input_data.append({
                                "audio"    : audio,
                                "audio_gt" : audio_gt,
                                "instruction": instruction,
                                "answer"   : reference,
                                "task_type": "MathQA"
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions, llm_text_inputs=None):

        data_with_model_predictions = []
        for idx, sample in enumerate(input_data):
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            if llm_text_inputs:
                new_sample['llm_text_input'] = llm_text_inputs[idx]
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, metrics=None):

        if metrics != 'acc':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'acc' for MathQA")
        
        predictions=[]
        references=[]
        for item in data_with_model_predictions:
            # prediction
            if item["model_prediction"] == None:
                item["model_prediction"] = "empty"
            else:
                model_prediction = utils.answer_clean('gsm8k', get_seperation_trigger('gsm8k'), item["model_prediction"])

            # answer
            answer = item["answer"]

            if not model_prediction: model_prediction = "empty"
            if not answer: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        details = []
        correct, wrong = 0, 0
        for prediction, reference in zip(predictions, references):
            if isinstance(reference, str):
                reference = [reference]
            if len(prediction) > 100: prediction=prediction[:100]
            if utils.compare_answer_with_groundtruth(prediction, *reference):
                correct += 1
            else:
                wrong += 1

            sample_details = {
                "reference" : reference,
                "prediction": prediction,
            }

            details.append(sample_details)

        return {"acc": correct / (correct + wrong), "details": details}
    


class spokenmqa_dataset_reasoning(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        # self.prompt   = math_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']
            audio_gt    = sample['context_transcript']
            reference   = sample['answer']['text']
            instruction = sample['instruction']['text']
            input_data.append({
                                "audio"    : audio,
                                "audio_gt" : audio_gt,
                                "instruction": instruction,
                                "answer"   : reference,
                                "task_type": "MathQA"
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions, llm_text_inputs=None):

        data_with_model_predictions = []
        for idx, sample in enumerate(input_data):
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            if llm_text_inputs:
                new_sample['llm_text_input'] = llm_text_inputs[idx]
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, metrics=None):

        if metrics != 'acc':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'acc' for MathQA")
        
        predictions=[]
        references=[]
        for item in data_with_model_predictions:
            # prediction
            if item["model_prediction"] == None:
                item["model_prediction"] = "empty"
            else:
                model_prediction = utils.answer_clean('gsm8k', get_seperation_trigger('gsm8k'), item["model_prediction"])

            # answer
            # gsm8k
            if "####" in item["answer"]:
                item["answer"] = utils.delete_extra_zero(item["answer"].split("#### ")[-1].replace(",", "")) 
            # svamp
            answer = utils.delete_extra_zero(item["answer"])

            if not model_prediction: model_prediction = "empty"
            if not answer: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        details = []
        correct, wrong = 0, 0
        for prediction, reference in zip(predictions, references):
            if isinstance(reference, str):
                reference = [reference]
            if len(prediction) > 100: prediction=prediction[:100]
            if utils.compare_answer_with_groundtruth(prediction, *reference):
                correct += 1
            else:
                wrong += 1

            sample_details = {
                "reference" : reference,
                "prediction": prediction,
            }

            details.append(sample_details)

        return {"acc": correct / (correct + wrong), "details": details}