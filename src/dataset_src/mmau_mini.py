import random
import logging
import pandas as pd

class mmau_mini_test_dataset(object):

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
            instruction = 'Question:\n' + sample['instruction'] + '\nChoices:\n' + " ".join(sample['choices'])
            reference   = sample['answer']

            input_data.append({
                                "audio"      : audio,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task"       : sample['other_attributes']['task'],
                                "task_type"  : "Audio-Understanding-Reasoning",
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
        
        questions   = [item["instruction"] for item in data_with_model_predictions]
        references  = [item["reference"] for item in data_with_model_predictions]
        predictions = [item["model_prediction"] for item in data_with_model_predictions]

        if metrics == 'llama3_70b_judge':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])

            # Grouping the data by 'task' and calculating the average rate_score for each task
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()
            
            return {'llama3_70b_judge': llama3_70b_judge_results, "task_scores": task_scores, 'details': all_details}


        if metrics == 'string_match':
            choices = [item for item in self.raw_data['choices']]
            from dataset_src.eval_methods.string_match import mmau_string_match
            string_match_results, all_details = mmau_string_match([questions, references, predictions, choices])

            # Grouping the data by 'task' and calculating the average rate_score for each task
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()

            return {'string_match': string_match_results, 'task_scores': task_scores, 'details': all_details}


        # elif metrics == 'llama3_8b_judge':
        #     from dataset_src.eval_methods.eval_llama3_8b import llama3_8b_as_judge
        #     llama3_8b_judge_results = llama3_8b_as_judge("../prepared_models/Meta-Llama-3-8B-Instruct-hf", [questions, references, predictions])
        #     return {'llama3_8b_judge': llama3_8b_judge_results}
        
        # elif metrics == 'prometheus2_judge':
        #     from dataset_src.eval_methods.eval_prometheus2 import prometheus2_as_judge
        #     prometheus2_judge_results = prometheus2_as_judge("../prepared_models/prometheus-7b-v2.0", [questions, references, predictions])
        #     return {'prometheus2_judge': prometheus2_judge_results}
        
        elif metrics == 'gpt4o_judge':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
            gpt4o_judge_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])


            # Grouping the data by 'task' and calculating the average rate_score for each task
            for result, sample_other_attributes in zip(all_details, self.raw_data['other_attributes']):
                result['task'] = sample_other_attributes['task']
            df = pd.DataFrame(all_details)
            task_scores = df.groupby('task')['rate_score'].mean().to_dict()

            return {'gpt4o_judge': gpt4o_judge_results, "task_scores": task_scores, 'details': all_details}
        
        # elif metrics == 'gpt4o_judge_binary':
        #     from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
        #     gpt4o_judge_binary_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
        #     return {'gpt4o_judge_binary': gpt4o_judge_binary_results, 'details': all_details}

        else:
            raise ValueError("Invalid metrics: {}".format(metrics))
