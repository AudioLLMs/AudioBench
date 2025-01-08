


import torch

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
import librosa

from tqdm import tqdm
import logging
import examples.AudioGemma2.modules_audio_gemma2.inference as inference

from hydra import initialize, compose



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cp_path = "examples/AudioGemma2/checkpoints/gemma2_2b_datasets_0908_ASR_ST_SI_0916/model.safetensors"
config_path="../../examples/AudioGemma2/checkpoints/gemma2_2b_datasets_0908_ASR_ST_SI_0916"
config_name="training_arguments"

with initialize(config_path=config_path):
    cfg = compose(config_name=config_name)

def load_model(_cfg):
    
    model = inference.load_model(_cfg, cp_path)

    data_collator = inference.SpeechTextDataCollator(
        feature_extractor=model.speech_encoder.feature_extractor,
        llm_tokenizer=model.text_decoder.tokenizer,
        use_audio_encoder=False,
        is_train=False,
        asr_normalize_text=cfg.asr_normalize_text,
        use_system_prompt=cfg.model.text_decoder.use_system_prompt,
    )
    return model, data_collator

def audiogemma_2_singlish_model_loader(self):

    print("model loading", flush=True)
    model, data_collator = load_model(cfg)
    print("model loaded", flush=True)

    self.model = model
    self.model_arguments = cfg
    self.data_collator = data_collator




def audiogemma_2_singlish_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. For ASR task, we will chunk and infer separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):
            
            # if chunk is less than 1 second, pad it to 1 second
            if len(chunk) < sampling_rate:
                chunk = np.pad(chunk, (0, sampling_rate - len(chunk)), 'constant', constant_values=(0, 0))
            
            audio_array = chunk

            samples = [{
                'audio'      : audio_array,
                'instruction': input["text"],
            }]
            batch = self.data_collator(samples)

            batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
                else v for k, v in batch.items()}
            for k, v in batch["speech_contexts_features"].items():
                v = v.to("cuda")
                
            with torch.no_grad():
                output = self.model.generate(**batch, synced_gpu=False)[0]

            model_predictions.append(output)
        
        output = ' '.join(model_predictions)

    else:
        
        # If audio duration is less than 1 second, we will pad the audio to 1 second
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant', constant_values=(0, 0))

        # For other tasks, if audio duration is more than 30 seconds, we will take first 30 seconds
        elif audio_duration > 30:
            logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
            audio_array = audio_array[:30 * sampling_rate]
            
        samples = [{
            'audio'      : audio_array,
            'instruction': input["text"],
        }]
        batch = self.data_collator(samples)
        batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
            else v for k, v in batch.items()}
        for k, v in batch["speech_contexts_features"].items():
            v = v.to("cuda")

        with torch.no_grad():
            output = self.model.generate(**batch, synced_gpu=False)[0]

    return output


