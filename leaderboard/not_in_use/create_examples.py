from datasets import Dataset, Value, Audio
import datasets

import os
import random
from tqdm import tqdm
import glob
import json
import pandas as pd
import soundfile as sf

# examples for "Automatic Speech Recognition"

def create_samples(dataset_lists, folder_path): 
    
    log_path = "/home/Collaborative_Projects/temp_transfer/audiobench/log/*"
    
    for data in tqdm(dataset_lists):
        
        if not os.path.exists(f"./examples/{folder_path}/{data}"):
            print(f"Create samples for {data} in {folder_path}")
            os.makedirs(f"./examples/{folder_path}/{data}")
        elif os.listdir(f"./examples/{folder_path}/{data}"):
            print(f"{data} in {folder_path} has examples")
            continue
        
        # if data == 'Spoken-Squad-v1':
        #     name = 'spoken_squad_test'
        # elif data == 'Tedlium3-Long-form-Test':
        #     name = 'tedlium3_long_form_test'

        if data == 'CN-College-Listen-MCQ-Test':
            name = 'cn_college_listen_test'
        elif data == 'DREAM-TTS-MCQ-Test':
            name = 'dream_tts_test'
        elif data == 'OpenHermes-Audio-Test':
            name = 'openhermes_instruction_test'
        elif data == "Clotho-AQA-Test":
            name = "clotho_asqa_test"
        else:
            name = data.lower().replace('-','_')


        if name in ['librispeech_test_clean', 'librispeech_test_other', 'tedlium3_test', 'tedlium3_long_form_test', 
                    'peoples_speech_test', 'common_voice_15_en_test', 'gigaspeech_test', "clotho_asqa_test",
                    'covost2_ta_en_test']:
            dataset = datasets.load_from_disk(f"/home/Collaborative_Projects/AudioBench-Related/AudioBench_private/data/{name}_v2")
        
        elif name in ['aishell_asr_zh_test', 'spoken_squad_test',
                      'covost2_en_id_test', 'covost2_en_zh_test', 'covost2_en_ta_test', 
                      'covost2_id_en_test', 'covost2_zh_en_test']:
            dataset = datasets.load_from_disk(f"/home/Collaborative_Projects/AudioBench-Related/AudioBench_private/data/{name}_v1")
        
        elif name in ['audiocaps_qa_test', 'wavcaps_qa_test']:
            dataset = datasets.load_from_disk(f"/home/Collaborative_Projects/AudioBench-Related/AudioBench_private/data/{name}_v3")
        
        else:
            dataset = datasets.load_from_disk(f"/home/Collaborative_Projects/AudioBench-Related/AudioBench_private/data/{name}")
        
        samples_index = random.sample(range(len(dataset)), 3)
        samples = dataset.select(samples_index)
        samples_df = pd.DataFrame(samples)
        
        log_name = data.lower().replace('-','_')
        for model_folder in glob.glob(log_path):
            
            if model_folder.split('/')[-1] in ['test_temp', 'merlion_v1']:
                continue
            
            log = []
            
            try:
                child_file = glob.glob(model_folder + f'/{log_name}.json')[0]
            except:
                continue
            
            with open(child_file) as f:
                file = json.load(f)
            
            try:
                for index in samples_index:
                    item  = file[index]
                    log.append(item)
            except:
                import pdb
                pdb.set_trace()
                continue
            
            samples_df[model_folder.split('/')[-1]] = log
        
        
        processed_samples = Dataset.from_pandas(samples_df)

        processed_samples = processed_samples.cast_column("context", 
                                                          {"text": Value(dtype='string'), 
                                                           "audio": Audio(sampling_rate=16000, decode=True),
                                                           },
                                                           )
        
        processed_samples.save_to_disk(f"./examples/{folder_path}/{data}")
        sf.write(f'./examples/{folder_path}/{data}/sample_0.wav', processed_samples[0]['context']['audio']['array'], samplerate=16000)
        sf.write(f'./examples/{folder_path}/{data}/sample_1.wav', processed_samples[1]['context']['audio']['array'], samplerate=16000)
        sf.write(f'./examples/{folder_path}/{data}/sample_2.wav', processed_samples[2]['context']['audio']['array'], samplerate=16000)

        

if __name__ == '__main__':
    
    asr_datasets = ['LibriSpeech-Test-Clean', 'LibriSpeech-Test-Other', 'Common-Voice-15-En-Test', 
                    'Peoples-Speech-Test', 'GigaSpeech-Test', 
                    #'Earnings21-Test', 
                    #'Earnings22-Test', 'Tedlium3-Test', 'Tedlium3-Long-form-Test', 
                    'IMDA-Part1-ASR-Test', 'IMDA-Part2-ASR-Test'
                    ]
    
    SQA_datasets = ['CN-College-Listen-MCQ-Test', 'DREAM-TTS-MCQ-Test','SLUE-P2-SQA5-Test', 
                    'Public-SG-Speech-QA-Test', 'Spoken-Squad-Test']
    
    SI_datasets = ['OpenHermes-Audio-Test', 'ALPACA-Audio-Test']
    
    AC_datasets = ['WavCaps-Test', 'AudioCaps-Test']
    
    ASQA_datasets = ['Clotho-AQA-Test', 'WavCaps-QA-Test', 'AudioCaps-QA-Test']
    
    AR_datasets = ['VoxCeleb-Accent-Test']
    
    GR_datasets = ['VoxCeleb-Gender-Test', 'IEMOCAP-Gender-Test']
    
    ER_datasets = ['IEMOCAP-Emotion-Test', 'MELD-Sentiment-Test', 'MELD-Emotion-Test']

    ST_datasets = ['Covost2-EN-ID-test', 
                    'Covost2-EN-ZH-test',
                    'Covost2-EN-TA-test', 
                    'Covost2-ID-EN-test', 
                    'Covost2-ZH-EN-test', 
                    'Covost2-TA-EN-test']
    
    CN_ASR_datasets = ['Aishell-ASR-ZH-Test']

    create_samples(asr_datasets, 'ASR')
    create_samples(SQA_datasets, 'SQA')
    create_samples(SI_datasets, 'SI')
    create_samples(AC_datasets, 'AC')
    create_samples(ASQA_datasets, 'AQA')
    create_samples(AR_datasets, 'AR')
    create_samples(GR_datasets, 'GR')
    create_samples(ER_datasets, 'ER')
    create_samples(ST_datasets, 'ST')
    create_samples(CN_ASR_datasets, 'CNASR')
