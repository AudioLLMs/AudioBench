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

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "Qwen/Qwen-Audio-Chat"

def qwen_audio_chat_model_loader(self):

    self.tokenizer               = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    self.model                   = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Model loaded: {}".format(model_path))




def post_process_qwen_asr(model_output):
    
    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    if ':"' in model_output:
        model_output = '"' + model_output.split(':"')[1]
    elif ': "' in model_output:
        model_output = '"' + model_output.split(': "')[1]

    # Find the longest match of ''
    match = re.search(r'"(.*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    return model_output


def qwen_audio_chat_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])

            # if len(audio_chunks) > 10:
            #     logger.info('More than 10 chunks. Taking first 10 chunks.')
            #     break
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            sf.write(audio_path.name, chunk, sampling_rate)

            query = self.tokenizer.from_list_format([
                {'audio': audio_path.name}, # Either a local path or an url
                {'text': input["instruction"]},
            ])
            response, history = self.model.chat(self.tokenizer, query=query, history=None)

            # Reprocess the results to get the output
            if input['task_type'] == 'ASR': response = post_process_qwen_asr(response)

            model_predictions.append(response)
        
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array[:30 * sampling_rate], sampling_rate)

        query = self.tokenizer.from_list_format([
            {'audio': audio_path.name}, # Either a local path or an url
            {'text': input["instruction"]},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)

        # Reprocess the results to get the output
        if input['task_type'] == 'ASR': response = post_process_qwen_asr(response)

        output = response
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)

        query = self.tokenizer.from_list_format([
            {'audio': audio_path.name}, # Either a local path or an url
            {'text': input["instruction"]},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        
        # Reprocess the results to get the output
        if input['task_type'] == 'ASR': response = post_process_qwen_asr(response)

        output = response



    return output

