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

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

whisper_model_path = "./examples/huayun_whisper_local_cs"
# whisper_model_path = "openai/whisper-large-v3"


def huayun_whisper_local_cs_model_loader(self):

    self.whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map="auto")
    self.whisper_processor = AutoProcessor.from_pretrained(whisper_model_path)
    self.whisper_pipe      = pipeline(
                    "automatic-speech-recognition",
                    model=self.whisper_model,
                    tokenizer=self.whisper_processor.tokenizer,
                    feature_extractor=self.whisper_processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    return_timestamps=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
    self.whisper_model.eval()

    logging.info(f"Model loaded from {whisper_model_path}.")


def huayun_whisper_local_cs_model_generation(self, sample):

    whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "en"})['text'].strip()
    
    if sample['task_type'] == "ASR":
        return whisper_output

    else:
        raise ValueError(f"Invalid task_type: {sample['task_type']}")
    