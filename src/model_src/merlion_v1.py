#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, June 24th 2024, 10:21:26 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('examples/merlion_v1/multimodal_trainer')

import logging
import numpy as np

import torch
from modules.models.speech_text_model import SpeechTextMultimodalModel
from modules.utils.model_args import TrainingArguments
from transformers import HfArgumentParser
from safetensors.torch import load_file
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
import torch.distributed as dist
import librosa


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg_json_file          = "examples/merlion_v1/ss_model/best/training_arguments.json"
model_safetensors_file = "examples/merlion_v1/ss_model/best/model_restored.safetensors"


from tqdm import trange, tqdm




# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

@dataclass
class DataCollator:
    speech_processor: Any
    llm_tokenizer: Any
    use_audio_encoder: bool
    is_train: bool = True
    asr_normalize_text: bool = False

    def convert_to_chat_template(self, instruction):

        conversation = [
            {
                "role": "system",
                "content": "You are a speech-text multimodal assistant. You would be given some speech tokens and text prompt tokens. Your job is to answer the question in the text prompt based on the context in speech tokens.",
            },
            {
                "role": "user",
                "content": f"Given the following speech tokens: \n<speech><SpeechHere></speech>\n\n{instruction}",
            },
        ]

        instruction_in_llm_template = self.llm_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False if self.is_train else True,
        )

        return instruction_in_llm_template
    

    def split_text_prompts(self, instructions):

        prompts_left = []
        prompts_right = []

        for instruction in instructions:
            instruction_in_llm_template = self.convert_to_chat_template(
                instruction=instruction
            )

            prompt_left, prompt_right = instruction_in_llm_template.split(
                "<SpeechHere>")

            prompts_left.append(prompt_left)
            prompts_right.append(prompt_right)

        return prompts_left, prompts_right

    def tokenize_left_prompts(self, prompts):
        return self.llm_tokenizer(
            text=prompts,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
            padding=True,
            truncation="longest_first",
            max_length=256,
        )

    def tokenize_right_prompts(self, prompts):
        if "zephyr-7b-beta" in self.llm_tokenizer.name_or_path:
            add_special_tokens = True
        else:
            add_special_tokens = False

        self.llm_tokenizer.add_bos_token = False
        prompts = self.llm_tokenizer(
            text                  = prompts,
            return_tensors        = "pt",
            add_special_tokens    = add_special_tokens,
            return_attention_mask = True,
            return_token_type_ids = True,
            padding               = True,
            truncation            = "longest_first",
            max_length            = 256,
        )
        self.llm_tokenizer.add_bos_token = True
        return prompts

    def __call__(self, examples: List) -> Dict[str, torch.Tensor]:

        speech_contexts = [example["audio"] for example in examples]
        speech_contexts_features = self.speech_processor.feature_extractor(
            speech_contexts,
            return_tensors="pt",
            sampling_rate=16000)

        instructions = [example["instruction"] for example in examples]

        text_prompts_left, text_prompts_right = self.split_text_prompts(
            instructions)

        text_prompts_left = self.tokenize_left_prompts(text_prompts_left)
        text_prompts_right = self.tokenize_right_prompts(text_prompts_right)

        return {
            "speech_contexts_features": speech_contexts_features,
            "text_prompts_left": text_prompts_left,
            "text_prompts_right": text_prompts_right,
            "audio_contexts_features": torch.zeros(5),
            "audio_contexts_mask": torch.zeros(5),
        }

def merlion_v1_model_loader(self):

    print("model loading", flush=True)
    parser          = HfArgumentParser(TrainingArguments)
    model_arguments = parser.parse_json_file(arg_json_file)[0]
    model           = SpeechTextMultimodalModel(**model_arguments.__dict__).to(device).eval()
    model.load_state_dict(load_file(model_safetensors_file))
    print("model loaded", flush=True)

    self.model           = model
    self.model_arguments = model_arguments
    self.data_collator = DataCollator(
        speech_processor   = model.get_speech_processor(),
        llm_tokenizer      = model.get_llm_tokenizer(),
        use_audio_encoder  = False,
        is_train           = False,
        asr_normalize_text = model_arguments.asr_normalize_text,
    )


def merlion_v1_model_generation(self, input):

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
            batch = {k: v.to(device) for k, v in batch.items()}
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
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = self.model.generate(**batch, synced_gpu=False)[0]

    return output


