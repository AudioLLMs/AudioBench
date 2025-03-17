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
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "SeaLLMs/SeaLLMs-Audio-7B"

def seallms_audio_7b_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(model_path)
    self.model = Qwen2AudioForConditionalGeneration.from_pretrained("SeaLLMs/SeaLLMs-Audio-7B", device_map="auto")
    #print("model.config._attn_implementation:", self.model.config._attn_implementation)
    #self.generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
    logger.info("Model loaded: {}".format(model_path))


def response_to_audio(conversation, model=None, processor=None):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if ele['audio_url'] != None:
                        audios.append(librosa.load(
                            ele['audio_url'], 
                            sr=processor.feature_extractor.sampling_rate)[0]
                        )
    if audios != []:

        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True,sampling_rate=16000)
    else: 
        inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items() if v is not None}
    generate_ids = model.generate(**inputs, max_new_tokens=2048, temperature = 0, do_sample=False)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def do_sample_inference(self, audio_array, prompt):

    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
    sf.write(audio_path.name, audio_array, 16000)


    # audio = [audio_array, 16000]

    # inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
    # generate_ids = self.model.generate(
    #         **inputs,
    #         max_new_tokens=1000,
    #         generation_config=self.generation_config,
    #     )
    # generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    # response = self.processor.batch_decode(
    #         generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )[0]

    # Audio Analysis
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path.name},
            {"type": "text", "text": prompt},
        ]},
    ]

    response = response_to_audio(conversation, model=self.model, processor=self.processor)

    return response


def seallms_audio_7b_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    instruction    = input['instruction']
    audio_duration = len(audio_array) / sampling_rate
    prompt         = instruction

    # user_prompt      = '<|user|>'
    # assistant_prompt = '<|assistant|>'
    # prompt_suffix    = '<|end|>'
    # prompt = f'{user_prompt}<|audio_1|>{instruction}{prompt_suffix}{assistant_prompt}'


    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 40 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 40 seconds. Chunking and inferring separately.')
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

