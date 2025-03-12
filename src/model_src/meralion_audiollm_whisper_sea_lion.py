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

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

repo_id = "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION"

def meralion_audiollm_whisper_sea_lion_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(
    repo_id, 
    trust_remote_code=True,
    )
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo_id,
        use_safetensors=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    self.model.to("cuda")

    logger.info("Model loaded: {}".format(repo_id))



def do_sample_inference(self, audio_array, instruction):

    prompt = "Given the following audio context: <SpeechHere>\n\nText instruction: {instruction}"
    conversation = [
            {"role": "user", "content": prompt.format(instruction=instruction)}
        ]

    chat_prompt = self.processor.tokenizer.apply_chat_template(
                conversation          = conversation,
                tokenize              = False,
                add_generation_prompt = True
            )

    inputs = self.processor(text=chat_prompt, audios=audio_array)

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to('cuda')
        if inputs[key].dtype is torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)

    model_outputs = self.model.generate(**inputs, max_new_tokens=228)
    generated_ids = model_outputs[:, inputs['input_ids'].size(1):]
    response      = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def meralion_audiollm_whisper_sea_lion_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    instruction = input["instruction"]

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

