#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 11:58:08 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
#
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import logging
import os

# add parent directory to sys.path
import sys
sys.path.append('.')

from datasets import load_dataset

from dataset_src.cn_college_listen_test import cn_college_listen_test_dataset
from dataset_src.slue_p2_sqa5_test import slue_p2_sqa5_test_dataset
from dataset_src.public_sg_speech_qa_test import public_sg_speech_qa_test_dataset


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


class Dataset(object):

    def __init__(self, dataset_name: str = "", number_of_samples: int = -1):

        self.dataset_name      = dataset_name
        self.number_of_samples = number_of_samples

        # Load dataset
        self.load_dataset()
        self.data_format()

    def load_dataset(self):

        logger.info("Loading dataset: {}".format(self.dataset_name))

        if   self.dataset_name == 'cn_college_listen_test': self.raw_data   = load_dataset("AudioLLMs/cn_college_listen_test")['test']
        elif self.dataset_name == 'slue_p2_sqa5_test': self.raw_data        = load_dataset("AudioLLMs/slue_p2_sqa5_test")['test']
        elif self.dataset_name == 'public_sg_speech_qa_test': self.raw_data = load_dataset("AudioLLMs/public_sg_speech_qa_test")['test']
        
        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
        logger.info("= = "*20)


    def data_format(self):

        if   self.dataset_name == 'cn_college_listen_test': self.dataset_processor   = cn_college_listen_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'slue_p2_sqa5_test': self.dataset_processor        = slue_p2_sqa5_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'public_sg_speech_qa_test': self.dataset_processor = public_sg_speech_qa_test_dataset(self.raw_data, self.number_of_samples)

        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        self.input_data = self.dataset_processor.prepare_model_input()

