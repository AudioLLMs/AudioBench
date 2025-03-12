import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import io
import os  
import base64
from openai import AzureOpenAI  
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  

import requests

import tempfile
import soundfile as sf



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


def gpt_4o_audio_model_loader(self):

    endpoint         = os.getenv("ENDPOINT_URL", "https://aoai-i2r-test-001.openai.azure.com/")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
    self.deployment  = os.getenv("DEPLOYMENT_NAME", "gpt-4o-audio-preview")

    # Initialize Azure OpenAI Service client with key-based authentication    
    self.client = AzureOpenAI(  
            azure_endpoint = endpoint,
            api_key        = subscription_key,
            api_version    = "2024-11-01-preview",
        )

    logger.info("Model loaded")


def do_sample_inference(self, audio_array, instruction, sampling_rate=16000):

    # Create an in-memory buffer
    buffer = io.BytesIO()

    # Write the WAV data to the buffer
    sf.write(buffer, audio_array, sampling_rate, format='WAV')

    # Get the byte data from buffer
    wav_data = buffer.getvalue()
    
    # Encode to Base64
    encoded_string = base64.b64encode(wav_data).decode('utf-8')

    chat_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        }
    ] 
        
    # Include speech result if speech is enabled  
    messages = chat_prompt 

    completion = self.client.chat.completions.create(  
            model             = self.deployment,
            messages          = messages,
            max_tokens        = 5000,
            temperature       = 0.7,
            top_p             = 0.95,
            frequency_penalty = 0,
            presence_penalty  = 0,
            stop              = None,
            stream            = False
        )  

    response = completion.choices[0].message.content

    return response



def gpt_4o_audio_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction    = input["instruction"]

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = [do_sample_inference(self, chunk_array, instruction) for chunk_array in tqdm(audio_chunks)]
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

        audio_array = audio_array[:30 * sampling_rate]
        output = do_sample_inference(self, audio_array, instruction)
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = do_sample_inference(self, audio_array, instruction)

    return output