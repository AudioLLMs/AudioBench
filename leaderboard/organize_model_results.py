import os
import json
import glob
from pathlib import Path
import pandas as pd



results_folder = '../log_for_all_models/'
model_names = [f.name for f in os.scandir(results_folder) if f.is_dir()]

# dataset - metrics - model performance
all_model_results = {}

for model_name in model_names:

    for filename in os.listdir(os.path.join(results_folder, model_name)):
        # skip file if not ends with '_score'
        if not filename.endswith('_score.json'):
            continue
        
        # get metrics
        if filename.endswith('_wer_score.json'):
            metric_name  = 'wer'
            dataset_name = filename.split(f'_{metric_name}_score.json')[0]
        elif filename.endswith('_llama3_70b_judge_score.json'):
            metric_name  = 'llama3_70b_judge'
            dataset_name = filename.split(f'_{metric_name}_score.json')[0]
        elif filename.endswith('_meteor_score.json'):
            metric_name  = 'meteor'
            dataset_name = filename.split(f'_{metric_name}_score.json')[0]
        elif filename.endswith('_bleu_score.json'):
            metric_name  = 'bleu'
            dataset_name = filename.split(f'_{metric_name}_score.json')[0]
        elif filename.endswith('_string_match_score.json'):
            metric_name  = 'string_match'
            dataset_name = filename.split(f'_{metric_name}_score.json')[0]
        elif filename.endswith('_gpt4o_judge_score.json'):
            metric_name  = 'gpt4o_judge'
            dataset_name = filename.split(f'_{metric_name}_score.json')[0]
        else:
            print('======'*3)
            print(model_name)
            print(filename)
            print('======'*3)

        with open(os.path.join(results_folder, model_name, filename), 'r') as f:
            result = json.load(f)

        if dataset_name not in all_model_results: all_model_results[dataset_name] = {}
        if metric_name not in all_model_results[dataset_name]: all_model_results[dataset_name][metric_name] = {}
        if model_name not in all_model_results[dataset_name][metric_name]: all_model_results[dataset_name][metric_name][model_name] = {}
        
        if metric_name == 'llama3_70b_judge' or metric_name == 'gpt4o_judge':
            all_model_results[dataset_name][metric_name][model_name] = result[metric_name]['judge_score']
        else: 
            all_model_results[dataset_name][metric_name][model_name] = result[metric_name]

with open('organize_model_results.json', 'w') as f:
    json.dump(all_model_results, f, indent=4)

