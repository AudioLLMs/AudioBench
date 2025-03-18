import random
import logging

ac_instructions = [
    "Could you help me create an audio caption for the provided clip?",
    "Please help me generate an audio caption for the audio clip.",
    "Would you mind helping me produce an audio caption for the provided clip?",
    "Could you help me formulate an audio caption for this clip?",
    "Could you assist me in creating a caption for the audio clip?",
]

class wavcaps_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.prompt = ac_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']
            instruction = random.choice(self.prompt)
            reference   = sample['answer']
            input_data.append({
                                "audio"      : audio,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task_type"  : "AC"
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
        
            question         = item["instruction"]
            answer           = item["reference"]
            model_prediction = item["model_prediction"]

            questions.append(question)
            references.append(answer)
            predictions.append(model_prediction)

        if metrics == 'llama3_70b_judge':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge
            llama3_70b_judge_results, all_details = llama3_70b_as_judge("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
            return {'llama3_70b_judge': llama3_70b_judge_results, 'details': all_details}
        
        # elif metrics == 'llama3_8b_judge':
        #     from dataset_src.eval_methods.eval_llama3_8b import llama3_8b_as_judge
        #     llama3_8b_judge_results = llama3_8b_as_judge("../prepared_models/Meta-Llama-3-8B-Instruct-hf", [questions, references, predictions])
        #     return {'llama3_8b_judge': llama3_8b_judge_results}
        
        # elif metrics == 'prometheus2_judge':
        #     from dataset_src.eval_methods.eval_prometheus2 import prometheus2_as_judge
        #     prometheus2_judge_results = prometheus2_as_judge("../prepared_models/prometheus-7b-v2.0", [questions, references, predictions])
        #     return {'prometheus2_judge': prometheus2_judge_results}
        
        elif metrics == 'gpt4o_judge':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge
            gpt4o_judge_results, all_details = gpt4o_as_judge("", [questions, references, predictions])
            return {'gpt4o_judge': gpt4o_judge_results, 'details': all_details}
        
        # elif metrics == 'gpt4o_judge_binary':
        #     from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
        #     gpt4o_judge_binary_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
        #     return {'gpt4o_judge_binary': gpt4o_judge_binary_results, 'details': all_details}
        
        elif metrics == 'meteor':
            import evaluate
            meteor         = evaluate.load('meteor')
            meteor_results = meteor.compute(predictions=predictions, references=references)
            return {'meteor': float(meteor_results['meteor'])}

        else:
            raise ValueError("Invalid metrics: {}".format(metrics))