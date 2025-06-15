import random
import logging

from jiwer import compute_measures, wer

def basic_text_normalization(text):
    return text.strip().lower()

class audiollm_instruction_following_dataset(object):
    def __init__(self, raw_data, number_of_samples):
        if isinstance(raw_data, dict):
            raw_data = raw_data["train"]
            
        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        self.raw_data = raw_data
        logging.info('Number of samples: {}'.format(len(self.raw_data)))
    
    def prepare_model_input(self):
        input_data = []
        print("overall: ", self.raw_data)
        for sample in self.raw_data:
            print("sample here: ", sample)
            audio       = sample['context']
            instruction = sample['instruction']
            answer      = sample['answer']
            instruction_type = sample['instruction_type']
            rule_type = sample['rule']
            rule_target = sample['rule_content']
            input_data.append({
                "audio": audio,
                "text": instruction,
                "answer": answer,
                "dimension": instruction_type,
                "rule_type": rule_type,
                "rule_target": rule_target,
                "task_type": instruction_type
            })
        logging.info('\n= = = Sample Input = = =')
        logging.info(random.choice(input_data))
        logging.info('= = = = = = = = = = = =\n')
        return input_data

    def format_model_predictions(self, input_data, model_predictions):
        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            if "audio" in new_sample:
                del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, metrics=None): 
        if metrics == 'llama3_70b_judge_combined':
            questions = []
            predictions = []
            references = []
            dimensions = []
            rules = []
            rule_targets = []

            for item in data_with_model_predictions:

                question         = item["text"]
                answer           = item["answer"]
                model_prediction = item["model_prediction"]
                dimension = item['dimension']
                rule = item['rule_type']
                rule_target = item['rule_target']

                questions.append(question)
                references.append(answer)
                predictions.append(model_prediction)
                dimensions.append(dimension)
                rules.append(rule)
                rule_targets.append(rule_target)
            
            from dataset_src.eval_methods.eval_llama3_70b_combined import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions, dimensions, rules, rule_targets])
            return {'llama3_70b_judge_combined': llama3_70b_judge_results, 'details': all_details}
        else:
            raise ValueError(f"Unsupported metric: {metrics}. Supported metric: 'llama3_70b_combined', 'llama3_70b_judge_binary'")