

import argparse
import os
import sys

import logging

import librosa
import torch
import transformers
from omegaconf import OmegaConf
from safetensors.torch import load_file

from tqdm import tqdm
import numpy as np

# HOME='/home'

#HOME='/project'

#sys.path.append(f'{HOME}/AudioBench_private/examples/MERaLiON_AudioLLM_v0_5_v2')   
#from multimodal_trainer.modules.models.speech_text_model import SpeechTextMultimodalModel
#from multimodal_trainer.modules.utils.data.batch_processor import mds_speech_text_batch_preprocessor
#from multimodal_trainer.modules.utils.data.instruct_collator import InstructDataCollator


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



def meralion_audiollm_v1_model_loader(self):

    #self.model_path = f"{HOME}/AudioBench_private/examples/MERaLiON_AudioLLM_v0_5_v2/checkpoint"


    print("model loading", flush=True)

    model_cfg_file = os.path.join(self.model_path, "training_arguments.yaml")
    model_path     = os.path.join(self.model_path, "model.safetensors")
    cfg            = OmegaConf.load(model_cfg_file)
    model_cfg      = cfg.model

    try:
        model = init_model(SpeechTextMultimodalModel, model_cfg)
    except:
        if self.multimodal_trainer_path not in sys.path:
            sys.path.append(self.multimodal_trainer_path)
        from modules.models.speech_text_model import SpeechTextMultimodalModel
        from modules.utils.module_patches import custom_get_cache
        from modules.models import init_model
        transformers.GenerationMixin._get_cache = custom_get_cache
        model = init_model(SpeechTextMultimodalModel, model_cfg)

    
    print(f"Loading model state dict from {model_path}")
    model.load_state_dict(load_file(model_path))
    model.eval()
    model.to("cuda:0")

    print("model loaded", flush=True)
    
    if not "multimodal_trainer.modules.utils.data.instruct_collator" in sys.modules:
        if self.multimodal_trainer_path not in sys.path:
            sys.path.append(self.multimodal_trainer_path)
        from modules.utils.data.instruct_collator import InstructDataCollator
        from modules.utils.data import init_data_collator

    speech_feature_extractor = {
        name: speech_encoder.feature_extractor
        for name, speech_encoder in model.speech_encoder.items()
    }
    data_collator = init_data_collator(
        InstructDataCollator,
        cfg,
        llm_tokenizer=model.text_decoder.tokenizer,
        speech_feature_extractor=speech_feature_extractor,
        audio_feature_extractor=(
            None if not model_cfg.use_audio_encoder else model.audio_encoder.feature_extractor
        ),
        is_train=False
    )


    self.model = model
    self.data_collator = data_collator


def get_batch(examples, batch_size=2):
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def format_example(speech_array, instruction_text):
    # convert speech and text into mosaic format so that we can directly use the data collator
    return {
        "context_text"     : None,
        "context_audio"    : speech_array,
        "instruction_text" : instruction_text,
        "instruction_audio": [0],
        "answer_text"      : None,
        "answer_audio"     : [0],
    }


def generate_one_sample(self, audio_array, instruction_text):
    examples = [format_example(audio_array, instruction_text)]

    with torch.no_grad():
        for batch in get_batch(examples, batch_size=1):
            batch = self.data_collator(batch)
            try:
                batch = mds_speech_text_batch_preprocessor(batch)
            except:
                if self.multimodal_trainer_path not in sys.path:
                    sys.path.append(self.multimodal_trainer_path)
                from modules.utils.data.batch_processor import mds_speech_text_batch_preprocessor
                batch = mds_speech_text_batch_preprocessor(batch)
            return self.model.generate(**batch, synced_gpus=False)[0]

def meralion_audiollm_v1_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    instruction    = input["text"]
    audio_duration = len(audio_array) / sampling_rate

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. For ASR task, we will chunk and infer separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):

            input = {
                "audio": {
                    "array"        : chunk,
                    "sampling_rate": sampling_rate,
                },
                "text": instruction,
            }
            model_response = meralion_audiollm_v1_model_generation(self, input)
            model_predictions.append(model_response)
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
        
        output = generate_one_sample(self, audio_array, instruction)
            
    return output

