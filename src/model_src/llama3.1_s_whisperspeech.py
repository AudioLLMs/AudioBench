#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, April 19th 2024, 11:17:41 am
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
import logging
import numpy as np
import torch
import torchaudio
from whisperspeech.vq_stoks import RQBottleneckTransformer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

llama_model_path = "/home/root/BachVD/model_zoo/llama3.1-s-instruct-2024-08-19-epoch-1/"

def audio_to_sound_tokens(vq_model, audio_path, target_bandwidth=1.5, device="cuda"):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vq_model.encode_audio(wav.to(device))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
def llama3_1_s_model_loader(self):

    self.vq_model = RQBottleneckTransformer.load_model(
        "../whisper-vq-stoks-medium-en+pl-fixed.model"
    ).to(self.device)
    self.vq_model.ensure_whisper(self.device)

    self.llm_tokenizer           = AutoTokenizer.from_pretrained(llama_model_path)
    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
    self.llm_model               = AutoModelForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.bfloat16, device=self.device)
    self.llm_model.eval()

    logging.info(f"Model loaded from {llama_model_path}.")


def llama3_1_s_model_generation(self, sample):

    audio_token = audio_to_sound_tokens(self.vq_model, sample['audio'])
    audio_token_with_transcript = f"<|reserved_special_token_69|>{audio_token}"
    
    if sample['task_type'] == "ASR":
        return audio_token
    
    question      = sample['text']

    batch_input = [question]

    if sample['task_type'] == "SI":
        batch_input = [audio_token]

    batch_input_templated = []
    for sample in batch_input:    
        messages = [
            {"role": "user", "content": sample},
        ]
        sample_templated = self.llm_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
        batch_input_templated.append(sample_templated)

    batch_input = batch_input_templated

    encoded_batch        = self.llm_tokenizer(batch_input, return_tensors="pt").to(self.llm_model.device)
    generated_ids        = self.llm_model.generate(**encoded_batch, max_new_tokens=1024)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if '</s>' in decoded_batch_output:
        decoded_batch_output = decoded_batch_output.split('</s>')[0].strip()
    
    return decoded_batch_output

