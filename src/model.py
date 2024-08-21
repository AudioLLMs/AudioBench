#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, November 10th 2023, 12:25:19 pm
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
import logging
import torch

from model_src.whisper_large_v3_with_llama_3_8b_instruct import whisper_large_v3_with_llama_3_8b_instruct_model_loader, whisper_large_v3_with_llama_3_8b_instruct_model_generation
from model_src.salmonn_7b import salmonn_7b_model_loader, salmonn_7b_model_generation
from model_src.llama3_1_s_whisperspeech import llama3_1_s_model_loader, llama3_1_s_model_generation

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
class Model(object):

    def __init__(self, model_name_or_path):

        self.model_name = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.load_model()
        logger.info("Loaded model: {}".format(self.model_name))
        logger.info("= = "*20)


    def load_model(self):

        if self.model_name == "whisper_large_v3_with_llama_3_8b_instruct": whisper_large_v3_with_llama_3_8b_instruct_model_loader(self)
        elif self.model_name == "salmonn_7b": salmonn_7b_model_loader(self)
        elif self.model_name == "llama3.1-s-whisperspeech": llama3_1_s_model_loader(self)

        
        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))


    def generate(self, input):

        with torch.no_grad():
            if self.model_name == "whisper_large_v3_with_llama_3_8b_instruct": return whisper_large_v3_with_llama_3_8b_instruct_model_generation(self, input)
            elif self.model_name == "salmonn_7b": return salmonn_7b_model_generation(self, input)
            elif self.model_name == "llama3.1-s-whisperspeech": return llama3_1_s_model_generation(self, input)
            
            else:
                raise NotImplementedError("Model {} not implemented yet".format(self.model_name))

