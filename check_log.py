
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
    'ytb_asr_batch2': ['wer'],
    'seame_dev_man' : ['wer'],
    'seame_dev_sge' : ['wer'],

    'aishell_asr_zh_test'       : ['wer'],

    'cn_college_listen_mcq_test': ['llama3_70b_judge'],
    'slue_p2_sqa5_test'         : ['llama3_70b_judge'],
    'dream_tts_mcq_test'        : ['llama3_70b_judge'],
    'public_sg_speech_qa_test'  : ['llama3_70b_judge'],
    'spoken_squad_test'         : ['llama3_70b_judge'],
    'openhermes_audio_test'     : ['llama3_70b_judge'],
    'alpaca_audio_test'         : ['llama3_70b_judge'],
    'clotho_aqa_test'           : ['llama3_70b_judge'],
    'wavcaps_qa_test'           : ['llama3_70b_judge'],
    'audiocaps_qa_test'         : ['llama3_70b_judge'],
    'wavcaps_test'              : ['llama3_70b_judge', 'meteor'],
    'audiocaps_test'            : ['llama3_70b_judge', 'meteor'],
    'iemocap_emotion_test'      : ['llama3_70b_judge'],
    'meld_sentiment_test'       : ['llama3_70b_judge'],
    'meld_emotion_test'         : ['llama3_70b_judge'],
    'voxceleb_accent_test'      : ['llama3_70b_judge'],
    'voxceleb_gender_test'      : ['llama3_70b_judge'],
    'iemocap_gender_test'       : ['llama3_70b_judge'],
    'covost2_en_id_test'        : ['bleu'],
    'covost2_en_zh_test'        : ['bleu'],
    'covost2_en_ta_test'        : ['bleu'],
    'covost2_id_en_test'        : ['bleu'],
    'covost2_zh_en_test'        : ['bleu'],
    'covost2_ta_en_test'        : ['bleu'],
    
    'muchomusic_test'           : ['llama3_70b_judge'],
    
    'imda_part1_asr_test'    : ['wer'],
    'imda_part2_asr_test'    : ['wer'],
    'imda_part3_30s_asr_test': ['wer'],
    'imda_part4_30s_asr_test': ['wer'],
    'imda_part5_30s_asr_test': ['wer'],
    'imda_part6_30s_asr_test': ['wer'],

    'imda_part3_30s_sqa_human_test': ['llama3_70b_judge'],
    'imda_part4_30s_sqa_human_test': ['llama3_70b_judge'],
    'imda_part5_30s_sqa_human_test': ['llama3_70b_judge'],
    'imda_part6_30s_sqa_human_test': ['llama3_70b_judge'],
    
    'imda_part3_30s_ds_human_test': ['llama3_70b_judge'],
    'imda_part4_30s_ds_human_test': ['llama3_70b_judge'],
    'imda_part5_30s_ds_human_test': ['llama3_70b_judge'],
    'imda_part6_30s_ds_human_test': ['llama3_70b_judge'],

    'imda_ar_sentence'      : ['llama3_70b_judge'],
    'imda_ar_dialogue'      : ['llama3_70b_judge'],
    'imda_gr_sentence'      : ['llama3_70b_judge'],
    'imda_gr_dialogue'      : ['llama3_70b_judge'],

    'ytb_sqa_batch1': ['llama3_70b_judge'],
    'ytb_sds_batch1': ['llama3_70b_judge'],
    'ytb_pqa_batch1': ['llama3_70b_judge'],

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

        # For ASR models, only exam ASR
        if MODEL_NAME == 'whisper_large_v3' and 'wer' not in model_score_name:
            continue
        
        score_log_path = f"log/{MODEL_NAME}/{model_score_name}_score.json"

        if os.path.exists(score_log_path) == False:
            print(f"Error: {score_log_path} not found.")
            continue

        try:

            with open(score_log_path, 'r') as f:
                json_score = json.load(f)
        except:
            print(f"Error: Failed to load json file {score_log_path}.")
            continue

    print("=====================================================")

# python check_log.py 2>&1 | tee check_log.log