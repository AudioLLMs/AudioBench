import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "microsoft/Phi-4-multimodal-instruct"

def phi_4_multimodal_instruct_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).cuda()
    print("model.config._attn_implementation:", self.model.config._attn_implementation)
    self.generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
    logger.info("Model loaded: {}".format(model_path))



def do_sample_inference(self, audio_array, prompt):

    audio = [audio_array, 16000]

    inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
    generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return response


def phi_4_multimodal_instruct_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    instruction    = input['instruction']
    audio_duration = len(audio_array) / sampling_rate

    user_prompt      = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix    = '<|end|>'
    prompt = f'{user_prompt}<|audio_1|>{instruction}{prompt_suffix}{assistant_prompt}'


    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 40 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 40 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 40 * sampling_rate])
        
        model_predictions = [do_sample_inference(self, chunk_array, prompt) for chunk_array in tqdm(audio_chunks)]
        output = ' '.join(model_predictions)


    elif audio_duration > 40:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

        audio_array = audio_array[:40 * sampling_rate]
        output = do_sample_inference(self, audio_array, prompt)
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = do_sample_inference(self, audio_array, prompt)

    return output

