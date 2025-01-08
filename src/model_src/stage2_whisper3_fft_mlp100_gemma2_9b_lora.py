

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
HOME='/home/users/astar/ares/wangb1/scratch'

sys.path.append(f'{HOME}/workspaces_wb/AudioBench_private/examples/stage2_whisper3_fft_mlp100_gemma2_9b_lora')   
from multimodal_trainer.modules.models.speech_text_model import SpeechTextMultimodalModel
from multimodal_trainer.modules.utils.data.batch_processor import mds_speech_text_batch_preprocessor
from multimodal_trainer.modules.utils.data.instruct_collator import InstructDataCollator
from multimodal_trainer.modules.utils.module_patches import custom_get_cache

transformers.GenerationMixin._get_cache = custom_get_cache


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



def stage2_whisper3_fft_mlp100_gemma2_9b_lora_model_loader(self):

    self.model_path = f"{HOME}/workspaces_wb/AudioBench_private/examples/stage2_whisper3_fft_mlp100_gemma2_9b_lora/checkpoint"


    print("model loading", flush=True)

    model_cfg_file = os.path.join(self.model_path, "training_arguments.yaml")
    model_path     = os.path.join(self.model_path, "model.safetensors")
    model_cfg      = OmegaConf.load(model_cfg_file).model

    model = SpeechTextMultimodalModel(
        speech_model_path=model_cfg.speech_encoder.path,
        text_decoder_path=model_cfg.text_decoder.path,
        train_speech_encoder=model_cfg.speech_encoder.requires_grad,
        speech_audio_adapter_type=model_cfg.speech_audio_adapter.type,
        speech_mlp_scale_factor=model_cfg.speech_audio_adapter.mlp_scale_factor,
        train_text_decoder=model_cfg.text_decoder.requires_grad,
        use_lora=model_cfg.text_decoder.use_lora,
        lora_alpha=model_cfg.text_decoder.lora.alpha,
        lora_rank=model_cfg.text_decoder.lora.rank,
        lora_dropout=model_cfg.text_decoder.lora.dropout,
        model_cfg=model_cfg,
        # hf_authentication_token=hf_authentication_token
    )

    
    print(f"Loading model state dict from {model_path}")
    model.load_state_dict(load_file(model_path))
    model.eval()
    model.to("cuda:0")

    # model, data_collator = load_model(cfg)
    print("model loaded", flush=True)

    data_collator = InstructDataCollator(
        speech_feature_extractor=model.speech_encoder.feature_extractor,
        llm_tokenizer=model.text_decoder.tokenizer,
        # use_audio_encoder=False,
        is_train=False,
        use_system_prompt=model_cfg.text_decoder.use_system_prompt,
    )


    self.model = model
    self.data_collator = data_collator


    # self.model_arguments = cfg
    # self.data_collator = data_collator


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
            batch = mds_speech_text_batch_preprocessor(batch)
            return self.model.generate(**batch, synced_gpus=False)[0]

def stage2_whisper3_fft_mlp100_gemma2_9b_lora_model_generation(self, input):

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
            model_response = stage2_whisper3_fft_mlp100_gemma2_9b_lora_model_generation(self, input)
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









# breakpoint()

# return None

# import torch

# import numpy as np
# from dataclasses import dataclass
# from typing import Any, Dict, List
# import librosa

# from tqdm import tqdm
# import logging
# import examples.AudioGemma2.modules_audio_gemma2.inference as inference

# from hydra import initialize, compose



# # =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =









# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cp_path = "examples/AudioGemma2/checkpoints/gemma2_2b_datasets_0908_0916/model.safetensors"
# config_path="../../examples/AudioGemma2/checkpoints/gemma2_2b_datasets_0908_0916"
# config_name="training_arguments"

# with initialize(config_path=config_path):
#     cfg = compose(config_name=config_name)

# def load_model(_cfg):
    
#     model = inference.load_model(_cfg, cp_path)

#     data_collator = inference.SpeechTextDataCollator(
#         feature_extractor=model.speech_encoder.feature_extractor,
#         llm_tokenizer=model.text_decoder.tokenizer,
#         use_audio_encoder=False,
#         is_train=False,
#         asr_normalize_text=cfg.asr_normalize_text,
#         use_system_prompt=cfg.model.text_decoder.use_system_prompt,
#     )
#     return model, data_collator





# @dataclass
# class DataCollator:
#     speech_processor: Any
#     llm_tokenizer: Any
#     use_audio_encoder: bool
#     is_train: bool = True
#     asr_normalize_text: bool = False

#     def convert_to_chat_template(self, instruction):

#         conversation = [
#             {
#                 "role": "system",
#                 "content": "You are a speech-text multimodal assistant. You would be given some speech tokens and text prompt tokens. Your job is to answer the question in the text prompt based on the context in speech tokens.",
#             },
#             {
#                 "role": "user",
#                 "content": f"Given the following speech tokens: \n<speech><SpeechHere></speech>\n\n{instruction}",
#             },
#         ]

#         instruction_in_llm_template = self.llm_tokenizer.apply_chat_template(
#             conversation,
#             tokenize=False,
#             add_generation_prompt=False if self.is_train else True,
#         )

#         return instruction_in_llm_template

#     def split_text_prompts(self, instructions):

#         prompts_left = []
#         prompts_right = []

#         for instruction in instructions:
#             instruction_in_llm_template = self.convert_to_chat_template(
#                 instruction=instruction
#             )

#             prompt_left, prompt_right = instruction_in_llm_template.split(
#                 "<SpeechHere>")

#             prompts_left.append(prompt_left)
#             prompts_right.append(prompt_right)

#         return prompts_left, prompts_right

#     def tokenize_left_prompts(self, prompts):
#         return self.llm_tokenizer(
#             text=prompts,
#             return_tensors="pt",
#             add_special_tokens=False,
#             return_attention_mask=True,
#             padding=True,
#             truncation="longest_first",
#             max_length=256,
#         )

#     def tokenize_right_prompts(self, prompts):
#         if ("zephyr-7b-beta" in self.llm_tokenizer.name_or_path) or ("Phi" in self.llm_tokenizer.name_or_path):
#             add_special_tokens = True
#         else:
#             add_special_tokens = False

#         self.llm_tokenizer.add_bos_token = False
#         prompts = self.llm_tokenizer(
#             text                  = prompts,
#             return_tensors        = "pt",
#             add_special_tokens    = add_special_tokens,
#             return_attention_mask = True,
#             return_token_type_ids = True,
#             padding               = True,
#             truncation            = "longest_first",
#             max_length            = 256,
#         )
#         self.llm_tokenizer.add_bos_token = True
#         return prompts

#     def __call__(self, examples: List) -> Dict[str, torch.Tensor]:

#         speech_contexts = [example["audio"] for example in examples]
#         speech_contexts_features = {
#             processor_name: speech_processor(
#             [resample_fn(s) for s in speech_contexts],
#             return_tensors="pt", return_attention_mask=True, **processor_args
#         ) for processor_name, (speech_processor, processor_args, resample_fn) in self.speech_processor.items()
#         }

#         instructions = [example["instruction"] for example in examples]

#         text_prompts_left, text_prompts_right = self.split_text_prompts(
#             instructions)

#         text_prompts_left = self.tokenize_left_prompts(text_prompts_left)
#         text_prompts_right = self.tokenize_right_prompts(text_prompts_right)

#         return {
#             "speech_contexts_features": speech_contexts_features,
#             "text_prompts_left": text_prompts_left,
#             "text_prompts_right": text_prompts_right,
#             "text_prompts_right_wo_responses": text_prompts_right,  # not used in model
#             "audio_contexts_features": torch.zeros(5),
#             "audio_contexts_mask": torch.zeros(5),
#         }



# def meralion_audiollm_v1_lora_model_generation(self, input):

#     audio_array    = input["audio"]["array"]
#     sampling_rate  = input["audio"]["sampling_rate"]
#     audio_duration = len(audio_array) / sampling_rate

#     # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
#     if audio_duration > 30 and input['task_type'] == 'ASR':
#         logger.info('Audio duration is more than 30 seconds. For ASR task, we will chunk and infer separately.')
#         audio_chunks = []
#         for i in range(0, len(audio_array), 30 * sampling_rate):
#             audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
#         model_predictions = []
#         for chunk in tqdm(audio_chunks):
            
#             # if chunk is less than 1 second, pad it to 1 second
#             if len(chunk) < sampling_rate:
#                 chunk = np.pad(chunk, (0, sampling_rate - len(chunk)), 'constant', constant_values=(0, 0))
            
#             audio_array = chunk

#             samples = [{
#                 'audio'      : audio_array,
#                 'instruction': input["text"],
#             }]
#             batch = self.data_collator(samples)

#             batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
#                 else v for k, v in batch.items()}
#             for k, v in batch["speech_contexts_features"].items():
#                 v = v.to("cuda")
                
#             with torch.no_grad():
#                 output = self.model.generate(**batch, synced_gpu=False)[0]

#             model_predictions.append(output)
        
#         output = ' '.join(model_predictions)

#     else:
        
#         # If audio duration is less than 1 second, we will pad the audio to 1 second
#         if audio_duration < 1:
#             logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
#             audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant', constant_values=(0, 0))

#         # For other tasks, if audio duration is more than 30 seconds, we will take first 30 seconds
#         elif audio_duration > 30:
#             logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
#             audio_array = audio_array[:30 * sampling_rate]
            
#         samples = [{
#             'audio'      : audio_array,
#             'instruction': input["text"],
#         }]
#         batch = self.data_collator(samples)
#         batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
#             else v for k, v in batch.items()}
#         for k, v in batch["speech_contexts_features"].items():
#             v = v.to("cuda")

#         with torch.no_grad():
#             output = self.model.generate(**batch, synced_gpu=False)[0]

#     return output




