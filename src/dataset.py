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

# ASR
from dataset_src.librispeech_test_clean import librispeech_test_clean_dataset
from dataset_src.librispeech_test_other import librispeech_test_other_dataset
from dataset_src.common_voice_15_en_test import common_voice_15_en_test_dataset
from dataset_src.peoples_speech_test import peoples_speech_test_dataset
from dataset_src.gigaspeech_test import gigaspeech_test_dataset
from dataset_src.earnings21_test import earnings21_test_dataset
from dataset_src.earnings22_test import earnings22_test_dataset
from dataset_src.tedlium3_test import tedlium3_test_dataset
from dataset_src.tedlium3_long_form_test import tedlium3_long_form_test_dataset

# ASR-ZH
from dataset_src.aishell_asr_zh_test import aishell_asr_zh_test_dataset

# ST
from dataset_src.covost2_en_id_test import covost2_en_id_test_dataset
from dataset_src.covost2_en_zh_test import covost2_en_zh_test_dataset
from dataset_src.covost2_en_ta_test import covost2_en_ta_test_dataset
from dataset_src.covost2_id_en_test import covost2_id_en_test_dataset
from dataset_src.covost2_zh_en_test import covost2_zh_en_test_dataset
from dataset_src.covost2_ta_en_test import covost2_ta_en_test_dataset

# SQA
from dataset_src.cn_college_listen_test import cn_college_listen_test_dataset
from dataset_src.cn_college_listen_mcq_test import cn_college_listen_mcq_test_dataset
from dataset_src.slue_p2_sqa5_test import slue_p2_sqa5_test_dataset
from dataset_src.public_sg_speech_qa_test import public_sg_speech_qa_test_dataset
from dataset_src.dream_tts_test import dream_tts_test_dataset
from dataset_src.dream_tts_mcq_test import dream_tts_mcq_test_dataset
from dataset_src.spoken_squad_test import spoken_squad_test_dataset

# SI
from dataset_src.openhermes_audio_test import openhermes_audio_test_dataset
from dataset_src.alpaca_audio_test import alpaca_audio_test_dataset

# AC
from dataset_src.audiocaps_test import audiocaps_test_dataset
from dataset_src.wavcaps_test import wavcaps_test_dataset

# ASQA
from dataset_src.clotho_aqa_test import clotho_aqa_test_dataset
from dataset_src.audiocaps_qa_test import audiocaps_qa_test_dataset
from dataset_src.wavcaps_qa_test import wavcaps_qa_test_dataset

# AR
from dataset_src.voxceleb_accent_test import voxceleb_accent_test_dataset

# GR
from dataset_src.voxceleb_gender_test import voxceleb_gender_test_dataset
from dataset_src.iemocap_gender_test import iemocap_gender_test_dataset

# ER
from dataset_src.iemocap_emotion_test import iemocap_emotion_test_dataset
from dataset_src.meld_sentiment_test import meld_sentiment_test_dataset
from dataset_src.meld_emotion_test import meld_emotion_test_dataset

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
        elif self.dataset_name == 'cn_college_listen_mcq_test': self.raw_data = load_dataset("AudioLLMs/cn_college_listen_mcq_test")['test']
        elif self.dataset_name == 'slue_p2_sqa5_test': self.raw_data        = load_dataset("AudioLLMs/slue_p2_sqa5_test")['test']
        elif self.dataset_name == 'public_sg_speech_qa_test': self.raw_data = load_dataset("AudioLLMs/public_sg_speech_qa_test")['test']
        elif self.dataset_name == 'dream_tts_test': self.raw_data           = load_dataset("AudioLLMs/dream_tts_test")['test']
        elif self.dataset_name == 'dream_tts_mcq_test': self.raw_data       = load_dataset("AudioLLMs/dream_tts_mcq_test")['test']
        elif self.dataset_name == 'librispeech_test_clean': self.raw_data   = load_dataset("AudioLLMs/librispeech_test_clean_v2")['test']
        elif self.dataset_name == 'librispeech_test_other': self.raw_data   = load_dataset("AudioLLMs/librispeech_test_other_v2")['test']
        elif self.dataset_name == 'common_voice_15_en_test': self.raw_data  = load_dataset("AudioLLMs/common_voice_15_en_test_v2")['test']
        elif self.dataset_name == 'peoples_speech_test': self.raw_data      = load_dataset("AudioLLMs/peoples_speech_test_v2")['test']
        elif self.dataset_name == 'gigaspeech_test': self.raw_data          = load_dataset("AudioLLMs/gigaspeech_test_v2")['test']
        elif self.dataset_name == 'earnings21_test': self.raw_data          = load_dataset("AudioLLMs/earnings21_test")['test']
        elif self.dataset_name == 'earnings22_test': self.raw_data          = load_dataset("AudioLLMs/earnings22_test")['test']
        elif self.dataset_name == 'tedlium3_test': self.raw_data            = load_dataset("AudioLLMs/tedlium3_test_v2")['test']
        elif self.dataset_name == 'tedlium3_long_form_test': self.raw_data  = load_dataset("AudioLLMs/tedlium3_long_form_test_v2")['test']
        elif self.dataset_name == 'openhermes_audio_test': self.raw_data    = load_dataset("AudioLLMs/openhermes_instruction_test")['test']
        elif self.dataset_name == 'alpaca_audio_test': self.raw_data        = load_dataset("AudioLLMs/alpaca_audio_test")['test']
        elif self.dataset_name == 'audiocaps_test': self.raw_data           = load_dataset("AudioLLMs/audiocaps_test")['test']
        elif self.dataset_name == 'wavcaps_test': self.raw_data             = load_dataset("AudioLLMs/wavcaps_test")['test']
        elif self.dataset_name == 'clotho_aqa_test': self.raw_data          = load_dataset("AudioLLMs/clotho_aqa_test")['test']
        elif self.dataset_name == 'audiocaps_qa_test': self.raw_data        = load_dataset("AudioLLMs/audiocaps_qa_test_v3")['test']
        elif self.dataset_name == 'wavcaps_qa_test': self.raw_data          = load_dataset("AudioLLMs/wavcaps_qa_test_v3")['test']
        elif self.dataset_name == 'voxceleb_accent_test': self.raw_data     = load_dataset("AudioLLMs/voxceleb_accent_test")['test']
        elif self.dataset_name == 'voxceleb_gender_test': self.raw_data     = load_dataset("AudioLLMs/voxceleb_gender_test")['test']
        elif self.dataset_name == 'iemocap_gender_test': self.raw_data      = load_dataset("AudioLLMs/iemocap_gender_test")['test']
        elif self.dataset_name == 'iemocap_emotion_test': self.raw_data     = load_dataset("AudioLLMs/iemocap_emotion_test")['test']
        elif self.dataset_name == 'meld_sentiment_test': self.raw_data      = load_dataset("AudioLLMs/meld_sentiment_test")['test']
        elif self.dataset_name == 'meld_emotion_test': self.raw_data        = load_dataset("AudioLLMs/meld_emotion_test")['test']
        
        elif self.dataset_name == 'covost2_en_id_test': self.raw_data = load_dataset("AudioLLMs/covost2_en_id_test_v1")['test']
        elif self.dataset_name == 'covost2_en_zh_test': self.raw_data = load_dataset("AudioLLMs/covost2_en_zh_test_v1")['test']
        elif self.dataset_name == 'covost2_en_ta_test': self.raw_data = load_dataset("AudioLLMs/covost2_en_ta_test_v1")['test']
        elif self.dataset_name == 'covost2_id_en_test': self.raw_data = load_dataset("AudioLLMs/covost2_id_en_test_v1")['test']
        elif self.dataset_name == 'covost2_zh_en_test': self.raw_data = load_dataset("AudioLLMs/covost2_zh_en_test_v1")['test']
        elif self.dataset_name == 'covost2_ta_en_test': self.raw_data = load_dataset("AudioLLMs/covost2_ta_en_test_v2")['test']

        elif self.dataset_name == 'aishell_asr_zh_test': self.raw_data = load_dataset("AudioLLMs/aishell_asr_zh_test_v1")['test']
        elif self.dataset_name == 'spoken_squad_test': self.raw_data   = load_dataset("AudioLLMs/spoken_squad_test_v1")['test']

        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
        logger.info("= = "*20)


    def data_format(self):

        # if samples less than requested samples
        if len(self.raw_data) < self.number_of_samples:
            self.number_of_samples = len(self.raw_data)
            logger.info("Number of samples requested is more than available samples. Setting number of samples to {}".format(self.number_of_samples))

        if   self.dataset_name == 'cn_college_listen_test': self.dataset_processor     = cn_college_listen_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'cn_college_listen_mcq_test': self.dataset_processor = cn_college_listen_mcq_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'slue_p2_sqa5_test': self.dataset_processor          = slue_p2_sqa5_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'public_sg_speech_qa_test': self.dataset_processor   = public_sg_speech_qa_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'dream_tts_test': self.dataset_processor             = dream_tts_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'dream_tts_mcq_test': self.dataset_processor         = dream_tts_mcq_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'librispeech_test_clean': self.dataset_processor     = librispeech_test_clean_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'librispeech_test_other': self.dataset_processor     = librispeech_test_other_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'common_voice_15_en_test': self.dataset_processor    = common_voice_15_en_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'peoples_speech_test': self.dataset_processor        = peoples_speech_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'gigaspeech_test': self.dataset_processor            = gigaspeech_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'earnings21_test': self.dataset_processor            = earnings21_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'earnings22_test': self.dataset_processor            = earnings22_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'tedlium3_test': self.dataset_processor              = tedlium3_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'tedlium3_long_form_test': self.dataset_processor    = tedlium3_long_form_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'openhermes_audio_test': self.dataset_processor      = openhermes_audio_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'alpaca_audio_test': self.dataset_processor          = alpaca_audio_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'audiocaps_test': self.dataset_processor             = audiocaps_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'wavcaps_test': self.dataset_processor               = wavcaps_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'clotho_aqa_test': self.dataset_processor            = clotho_aqa_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'audiocaps_qa_test': self.dataset_processor          = audiocaps_qa_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'wavcaps_qa_test': self.dataset_processor            = wavcaps_qa_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'voxceleb_accent_test': self.dataset_processor       = voxceleb_accent_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'voxceleb_gender_test': self.dataset_processor       = voxceleb_gender_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'iemocap_gender_test': self.dataset_processor        = iemocap_gender_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'iemocap_emotion_test': self.dataset_processor       = iemocap_emotion_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'meld_sentiment_test': self.dataset_processor        = meld_sentiment_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'meld_emotion_test': self.dataset_processor          = meld_emotion_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_en_id_test': self.dataset_processor = covost2_en_id_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'covost2_en_zh_test': self.dataset_processor = covost2_en_zh_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'covost2_en_ta_test': self.dataset_processor = covost2_en_ta_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'covost2_id_en_test': self.dataset_processor = covost2_id_en_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'covost2_zh_en_test': self.dataset_processor = covost2_zh_en_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'covost2_ta_en_test': self.dataset_processor = covost2_ta_en_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'aishell_asr_zh_test': self.dataset_processor = aishell_asr_zh_test_dataset(self.raw_data, self.number_of_samples)
        elif self.dataset_name == 'spoken_squad_test': self.dataset_processor   = spoken_squad_test_dataset(self.raw_data, self.number_of_samples)


        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        self.input_data = self.dataset_processor.prepare_model_input()

