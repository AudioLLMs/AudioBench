
import os
import json


DATASETS_TO_CHECK = {
    'librispeech_test_clean' : ['wer'],
    'librispeech_test_other' : ['wer'],
    'common_voice_15_en_test': ['wer'],
    'peoples_speech_test'    : ['wer'],
    'gigaspeech_test'        : ['wer'],
    'tedlium3_test'          : ['wer'],
    'tedlium3_long_form_test': ['wer'],
    'earnings21_test'        : ['wer'],
    'earnings22_test'        : ['wer'],

    'cna_test'             : ['wer'],
    'idpc_test'            : ['wer'],
    'parliament_test'      : ['wer'],
    'ukusnews_test'        : ['wer'],
    'mediacorp_test'       : ['wer'],
    'idpc_short_test'      : ['wer'],
    'parliament_short_test': ['wer'],
    'ukusnews_short_test'  : ['wer'],
    'mediacorp_short_test' : ['wer'],

    'ytb_asr_batch1': ['wer'],
    'seame_dev_man' : ['wer'],
    'seame_dev_sge' : ['wer'],



    'aishell_asr_zh_test'       : ['wer'],

    'cn_college_listen_mcq_test': ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'slue_p2_sqa5_test'         : ['llama3_70b_judge', 'gpt4o_judge'],
    'dream_tts_mcq_test'        : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'public_sg_speech_qa_test'  : ['llama3_70b_judge', 'gpt4o_judge'],
    'spoken_squad_test'         : ['llama3_70b_judge', 'gpt4o_judge'],
    'openhermes_audio_test'     : ['llama3_70b_judge', 'gpt4o_judge'],
    'alpaca_audio_test'         : ['llama3_70b_judge', 'gpt4o_judge'],
    'clotho_aqa_test'           : ['llama3_70b_judge', 'gpt4o_judge'],
    'wavcaps_qa_test'           : ['llama3_70b_judge', 'gpt4o_judge'],
    'audiocaps_qa_test'         : ['llama3_70b_judge', 'gpt4o_judge'],
    'wavcaps_test'              : ['llama3_70b_judge', 'meteor', 'gpt4o_judge'],
    'audiocaps_test'            : ['llama3_70b_judge', 'meteor', 'gpt4o_judge'],
    'iemocap_emotion_test'      : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'meld_sentiment_test'       : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'meld_emotion_test'         : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'voxceleb_accent_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    'voxceleb_gender_test'      : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'iemocap_gender_test'       : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'covost2_en_id_test'        : ['bleu'],
    'covost2_en_zh_test'        : ['bleu'],
    'covost2_en_ta_test'        : ['bleu'],
    'covost2_id_en_test'        : ['bleu'],
    'covost2_zh_en_test'        : ['bleu'],
    'covost2_ta_en_test'        : ['bleu'],
    'muchomusic_test'           : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    
    
    # 'imda_part3_30s_sqa_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    # 'imda_part4_30s_sqa_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    # 'imda_part5_30s_sqa_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    # 'imda_part6_30s_sqa_test'      : ['llama3_70b_judge', 'gpt4o_judge'],

    # 'imda_part3_30s_ds_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    # 'imda_part4_30s_ds_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    # 'imda_part5_30s_ds_test'      : ['llama3_70b_judge', 'gpt4o_judge'],
    # 'imda_part6_30s_ds_test'      : ['llama3_70b_judge', 'gpt4o_judge'],

    'imda_part1_asr_test'    : ['wer'],
    'imda_part2_asr_test'    : ['wer'],
    'imda_part3_30s_asr_test': ['wer'],
    'imda_part4_30s_asr_test': ['wer'],
    'imda_part5_30s_asr_test': ['wer'],
    'imda_part6_30s_asr_test': ['wer'],

    'imda_part3_30s_sqa_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    'imda_part4_30s_sqa_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    'imda_part5_30s_sqa_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    'imda_part6_30s_sqa_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    
    'imda_part3_30s_ds_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    'imda_part4_30s_ds_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    'imda_part5_30s_ds_human_test': ['llama3_70b_judge', 'gpt4o_judge'],
    'imda_part6_30s_ds_human_test': ['llama3_70b_judge', 'gpt4o_judge'],

    'imda_ar_sentence'      : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'imda_ar_dialogue'      : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'imda_gr_sentence'      : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    'imda_gr_dialogue'      : ['llama3_70b_judge_binary', 'gpt4o_judge_binary'],
    
}

MODEL_SCORE_NAMES = []
for dataset_name, metric_names in DATASETS_TO_CHECK.items():
    for metric_name in metric_names:
        MODEL_SCORE_NAMES += [f"{dataset_name}_{metric_name}"]



MODEL_NAME_TO_CHECK = os.listdir('log')
# sort by model names
MODEL_NAME_TO_CHECK.sort()

for MODEL_NAME in MODEL_NAME_TO_CHECK:

    if MODEL_NAME == 'old_models' or MODEL_NAME == 'test_temp':
        continue

    print(f"Checking {MODEL_NAME}")


    for model_score_name in MODEL_SCORE_NAMES:

        if 'gpt4o' in model_score_name: # skip gpt4o
            continue
        
        #print(f"Checking {model_score_name}...")
        #output_log_path = f"log/{MODEL_NAME}/{model_score_name}.json"
        score_log_path = f"log/{MODEL_NAME}/{model_score_name}_score.json"

        #if os.path.exists(output_log_path) == False:
        #    print(f"Error: {output_log_path} not found.")
        #    continue
        if os.path.exists(score_log_path) == False:
            print(f"Error: {score_log_path} not found.")
            continue

        try:
            #with open(output_log_path, 'r') as f:
            #    json_log = json.load(f)
            with open(score_log_path, 'r') as f:
                json_score = json.load(f)
        except:
            #print(f"Error: Failed to load json file {output_log_path}, {score_log_path}.")
            print(f"Error: Failed to load json file {score_log_path}.")
            continue

        #print("ALL CLEAR!")
    print("=====================================================")

# python check_log.py 2>&1 | tee check_log.log