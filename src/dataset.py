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

        if self.dataset_name == 'cn_college_listen_test': 
            self.raw_data = load_dataset("AudioLLMs/cn_college_listen_test")['test']
        
        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
        logger.info("= = "*20)


    def data_format(self):

        if self.dataset_name == 'cn_college_listen_test': 
            self.dataset_processor = cn_college_listen_test_dataset(self.raw_data, self.number_of_samples)

        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        self.input_data = self.dataset_processor.prepare_model_input()

