import os
import json
import glob
from pathlib import Path
import pandas as pd



def create_df(dataset_names, category, metrics):

    df = pd.DataFrame(columns=['Model'] + dataset_names)

    for folder in child_folders:
        child_files = glob.glob(folder + '/*_score.json')
        dict = {'Model': folder.split('/')[-1]}
        
        for item in child_files:
            for dataset in dataset_names:
                if f'{dataset}_{metrics}_score.json' == item.split('/')[-1]:
                    
                    with open(item) as f:
                        file = json.load(f)

                    if metrics == 'wer':
                        dict[dataset] = file[metrics]
                    elif metrics == 'bleu':
                        dict[dataset] = file[metrics]
                    elif metrics == 'llama3_70b_judge':
                        dict[dataset] = file[metrics]['judge_score']
                    elif metrics == 'meteor':
                        dict[dataset] = file[metrics]
            
        df = df._append(dict, ignore_index = True)

    if not os.path.exists(f'./results_organized/{metrics}'):
        os.makedirs(f'./results_organized/{metrics}')
    # df.to_csv(f"./results_organized/{metrics}/{category}.csv", index = False)

    breakpoint()

    df.to_json(f"./results_organized/{metrics}/{category}.json")
    
    print(f'{category} Saved.')



if __name__ == "__main__":
    root = "../log_for_all_models/*"
    child_folders = glob.glob(root)
    
    ASR_datasets = [
                    'librispeech_test_clean', 
                    'librispeech_test_other', 
                    'common_voice_15_en_test', 
                    'peoples_speech_test', 
                    'gigaspeech_test', 
                    'earnings21_test', 
                    'earnings22_test', 
                    'tedlium3_test', 
                    'tedlium3_long_form_test', 
                    ]
    create_df(ASR_datasets, 'asr_english', 'wer')


    SQA_datasets = [
                      'slue_p2_sqa5_test',
                      'public_sg_speech_qa_test', 
                      'spoken_squad_test',
                      'cn_college_listen_mcq_test', 
                      'dream_tts_mcq_test',
                      ]
    create_df(SQA_datasets, 'sqa_english', 'llama3_70b_judge')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    SI_datasets = [
                    'openhermes_audio_test', 
                    'alpaca_audio_test',
                   ]
    create_df(SI_datasets, 'speech_instruction', 'llama3_70b_judge')




    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    AC_datasets = [
                    'audiocaps_test', 
                    'wavcaps_test',
                   ]
    create_df(AC_datasets, 'audio_captioning', 'llama3_70b_judge')
    create_df(AC_datasets, 'audio_captioning', 'meteor')





    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    ASQA_datasets = [
                     'clotho_aqa_test', 
                     'audiocaps_qa_test', 
                     'wavcaps_qa_test',
                     ]
    create_df(ASQA_datasets, 'audio_scene_question_answering', 'llama3_70b_judge')




    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    AR_datasets = [
                    'voxceleb_accent_test',
                    'imda_ar_sentence',
                    'imda_ar_dialogue',
                   ]
    create_df(AR_datasets, 'accent_recognition', 'llama3_70b_judge')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    GR_datasets = ['voxceleb_gender_test', 
                   'iemocap_gender_test',
                   'imda_gr_sentence',
                   'imda_gr_dialogue',
                   ]
    create_df(GR_datasets, 'gender_recognition', 'llama3_70b_judge')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    ER_datasets = ['iemocap_emotion_test', 
                   'meld_sentiment_test', 
                   'meld_emotion_test',
                   ]
    create_df(ER_datasets, 'emotion_recognition', 'llama3_70b_judge')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    ST_datasets = ['covost2_en_id_test', 
                   'covost2_en_zh_test', 
                   'covost2_en_ta_test', 
                   'covost2_id_en_test', 
                   'covost2_zh_en_test', 
                   'covost2_ta_en_test',
                   ]
    create_df(ST_datasets, 'st', 'bleu')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    CN_ASR_datasets = ['aishell_asr_zh_test',
                       ]
    create_df(CN_ASR_datasets, 'asr_mandarin', 'wer')




    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    singlish_asr_datasets = ['imda_part1_asr_test', 
                             'imda_part2_asr_test',
                             'imda_part3_30s_asr_test',
                             'imda_part4_30s_asr_test',
                             'imda_part5_30s_asr_test',
                             'imda_part6_30s_asr_test',
                             'seame_dev_man',
                             'seame_dev_sge',
                             ]
    create_df(singlish_asr_datasets, 'asr_singlish', 'wer')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    singlish_sqa_datasets = ['imda_part3_30s_sqa_human_test', 
                             'imda_part4_30s_sqa_human_test',
                             'imda_part5_30s_sqa_human_test',
                             'imda_part6_30s_sqa_human_test',
                             ]
    create_df(singlish_sqa_datasets, 'sqa_singlish', 'llama3_70b_judge')

    


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    music_mcq_datasets = ['muchomusic_test']
    create_df(music_mcq_datasets, 'music_understanding', 'llama3_70b_judge')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    singlish_sds_datasets = ['imda_part3_30s_ds_human_test', 
                             'imda_part4_30s_ds_human_test',
                             'imda_part5_30s_ds_human_test',
                             'imda_part6_30s_ds_human_test',
                             ]
    create_df(singlish_sds_datasets, 'sds_singlish', 'llama3_70b_judge')



    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =    
    under_development_wer_datasets = [
                                'cna_test',
                                'idpc_test',
                                'parliament_test',
                                'ukusnews_test',
                                'mediacorp_test',
                                'idpc_short_test',
                                'parliament_short_test',
                                'ukusnews_short_test',
                                'mediacorp_short_test',
                                'ytb_asr_batch1',
                                'ytb_asr_batch2',
                             ]
    create_df(under_development_wer_datasets, 'under_development_wer', 'wer')

    
    under_development_llama3_70b_judge_datasets = [
                                'ytb_sqa_batch1',
                                'ytb_sds_batch1',
                                'ytb_pqa_batch1',
                             ]
    create_df(under_development_llama3_70b_judge_datasets, 'under_development_llama3_70b_judge', 'llama3_70b_judge')


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =    
    create_df(['mmau_mini'], 'mmau_mini', 'llama3_70b_judge')








    




    






