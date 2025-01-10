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

        self.dataset_name = None
        self.model_name   = model_name_or_path
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

        self.load_model()
        logger.info("Loaded model: {}".format(self.model_name))
        logger.info("= = "*20)


    def load_model(self):

        if self.model_name == "cascade_whisper_large_v3_llama_3_8b_instruct": 
            from model_src.whisper_large_v3_with_llama_3_8b_instruct import whisper_large_v3_with_llama_3_8b_instruct_model_loader
            whisper_large_v3_with_llama_3_8b_instruct_model_loader(self)

        elif self.model_name == "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct":
            from model_src.cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_loader
            cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_loader(self)
        
        elif self.model_name == "temp_model_for_debugging_datasets":
            from model_src.temp_debug_datasets import temp_debug_datasets_model_loader
            temp_debug_datasets_model_loader(self)

        elif self.model_name == "huayun_whisper_local_cs":
            from model_src.huayun_whisper_local_cs import huayun_whisper_local_cs_model_loader
            huayun_whisper_local_cs_model_loader(self)

        elif self.model_name == "huayun_whisper_local_no_cs":
            from model_src.huayun_whisper_local_no_cs import huayun_whisper_local_no_cs_model_loader
            huayun_whisper_local_no_cs_model_loader(self)

        elif self.model_name == "xl_whisper_imda_v0_1":
            from model_src.xl_whisper_imda_v0_1 import xl_whisper_imda_v0_1_model_loader
            xl_whisper_imda_v0_1_model_loader(self)

        elif self.model_name == "original_whisper_large_v2":
            from model_src.original_whisper_large_v2 import original_whisper_large_v2_model_loader
            original_whisper_large_v2_model_loader(self)
        
        elif self.model_name == "Qwen2-Audio-7B-Instruct":
            from model_src.qwen2_audio_7b_instruct import qwen2_audio_7b_instruct_model_loader
            qwen2_audio_7b_instruct_model_loader(self)

        elif self.model_name == "MERaLiON_AudioLLM_v0_5": 
            from model_src.meralion_audiollm_v0_5 import meralion_audiollm_v0_5_model_loader
            meralion_audiollm_v0_5_model_loader(self)

        elif self.model_name == "MERaLiON_AudioLLM_v0_5_v2": 
            from model_src.meralion_audiollm_v0_5_v2 import meralion_audiollm_v0_5_v2_model_loader
            meralion_audiollm_v0_5_v2_model_loader(self)

        elif self.model_name == "MERaLiON_AudioLLM_v0_5_average5" or self.model_name == "MERaLiON_AudioLLM_v0_5_average5_better_asr":
            from model_src.MERaLiON_AudioLLM_v0_5_average5 import MERaLiON_AudioLLM_v0_5_average5_model_loader
            MERaLiON_AudioLLM_v0_5_average5_model_loader(self)

        elif self.model_name == "MERaLiON_AudioLLM_v1": 
            from model_src.meralion_audiollm_v1 import meralion_audiollm_v1_model_loader
            meralion_audiollm_v1_model_loader(self)

        elif self.model_name == "SALMONN_7B":
            from model_src.salmonn_7b import salmonn_7b_model_loader
            salmonn_7b_model_loader(self)

        elif self.model_name == 'WavLLM_fairseq': 
            from model_src.wavllm_fairseq import wavllm_fairseq_model_loader
            wavllm_fairseq_model_loader(self)

        elif self.model_name == 'Qwen-Audio-Chat':
            from model_src.qwen_audio_chat import qwen_audio_chat_model_loader
            qwen_audio_chat_model_loader(self)

        elif self.model_name.startswith("AudioLLM_IMDA_"):
            from model_src.audiollm_imda import audiollm_imda_model_loader
            audiollm_imda_model_loader(self)

        elif self.model_name == 'MERaLiON-AudioLLM-Whisper-SEA-LION':
            from model_src.meralion_audiollm_whisper_sea_lion import meralion_audiollm_whisper_sea_lion_model_loader
            meralion_audiollm_whisper_sea_lion_model_loader(self)

        elif self.model_name == 'gemini-1.5-flash':
            from model_src.gemini_1_5_flash import gemini_1_5_flash_model_loader
            gemini_1_5_flash_model_loader(self)







        elif self.model_name == 'test_temp': test_temp_model_loader(self)
        elif self.model_name == "merlion_v1": merlion_v1_model_loader(self)

        # MOWE-Audio
        elif self.model_name == "mowe_audio": mowe_audio_model_loader(self)
        elif self.model_name == "multitask-subsetv2:whisper_specaugment+seqcnn8+lora:f3epoch:run2.0:4gpu": temp_mowe_audio_11_24_model_loader(self)

        elif self.model_name == "AudioGemma2_v1": audiogemma_2_v1_model_loader(self)
        elif self.model_name == "audiogemma_2_singlish": audiogemma_2_singlish_model_loader(self)
        elif self.model_name == "meralion_audiollm_v1_lora": meralion_audiollm_v1_lora_model_loader(self)
        elif self.model_name == "meralion_audiollm_v1_mse": meralion_audiollm_v1_mse_model_loader(self)
        elif self.model_name == "stage2_whisper3_fft_mlp100_gemma2_9b_lora": stage2_whisper3_fft_mlp100_gemma2_9b_lora_model_loader(self)
        elif self.model_name == 'stage2_only_whisper3_fft_mlp100_sealion3_9b_lora': stage2_only_whisper3_fft_mlp100_sealion3_9b_lora_model_loader(self)
        elif self.model_name == 'audiollm_imda': audiollm_imda_model_loader(self)

        elif 'AudioGemma_IMDA_' in self.model_name:
            audiogemma_imda_model_loader(self)
        

        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))


    def generate(self, input):

        with torch.no_grad():
            if self.model_name == "cascade_whisper_large_v3_llama_3_8b_instruct": 
                from model_src.whisper_large_v3_with_llama_3_8b_instruct import whisper_large_v3_with_llama_3_8b_instruct_model_generation
                return whisper_large_v3_with_llama_3_8b_instruct_model_generation(self, input)
            
            elif self.model_name == "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct":
                from model_src.cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_generation
                return cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_generation(self, input)
            
            elif self.model_name == "temp_model_for_debugging_datasets":
                from model_src.temp_debug_datasets import temp_debug_datasets_model_generation
                return temp_debug_datasets_model_generation(self, input)

            elif self.model_name == "huayun_whisper_local_cs":
                from model_src.huayun_whisper_local_cs import huayun_whisper_local_cs_model_generation
                return huayun_whisper_local_cs_model_generation(self, input)

            elif self.model_name == "huayun_whisper_local_no_cs":
                from model_src.huayun_whisper_local_no_cs import huayun_whisper_local_no_cs_model_generation
                return huayun_whisper_local_no_cs_model_generation(self, input)

            elif self.model_name == "xl_whisper_imda_v0_1":
                from model_src.xl_whisper_imda_v0_1 import xl_whisper_imda_v0_1_model_generation
                return xl_whisper_imda_v0_1_model_generation(self, input)

            elif self.model_name == "original_whisper_large_v2":
                from model_src.original_whisper_large_v2 import original_whisper_large_v2_model_generation
                return original_whisper_large_v2_model_generation(self, input)

            elif self.model_name == "Qwen2-Audio-7B-Instruct":
                from model_src.qwen2_audio_7b_instruct import qwen2_audio_7b_instruct_model_generation
                return qwen2_audio_7b_instruct_model_generation(self, input)

            elif self.model_name == "MERaLiON_AudioLLM_v0_5": 
                from model_src.meralion_audiollm_v0_5 import meralion_audiollm_v0_5_model_generation
                return meralion_audiollm_v0_5_model_generation(self, input)

            elif self.model_name == "MERaLiON_AudioLLM_v0_5_v2":
                from model_src.meralion_audiollm_v0_5_v2 import meralion_audiollm_v0_5_v2_model_generation
                return meralion_audiollm_v0_5_v2_model_generation(self, input)

            elif self.model_name == "MERaLiON_AudioLLM_v0_5_average5" or self.model_name == "MERaLiON_AudioLLM_v0_5_average5_better_asr":
                from model_src.MERaLiON_AudioLLM_v0_5_average5 import MERaLiON_AudioLLM_v0_5_average5_model_generation
                return MERaLiON_AudioLLM_v0_5_average5_model_generation(self, input)

            elif self.model_name == "MERaLiON_AudioLLM_v1": 
                from model_src.meralion_audiollm_v1 import meralion_audiollm_v1_model_generation
                return meralion_audiollm_v1_model_generation(self, input)
            
            elif self.model_name == "SALMONN_7B":
                from model_src.salmonn_7b import salmonn_7b_model_generation
                return salmonn_7b_model_generation(self, input)
            
            elif self.model_name == "WavLLM_fairseq":
                from model_src.wavllm_fairseq import wavllm_fairseq_model_generation
                return wavllm_fairseq_model_generation(self, input)
            
            elif self.model_name == "Qwen-Audio-Chat":
                from model_src.qwen_audio_chat import qwen_audio_chat_model_generation
                return qwen_audio_chat_model_generation(self, input)
            
            elif self.model_name.startswith("AudioLLM_IMDA_"):
                from model_src.audiollm_imda import audiollm_imda_model_generation
                return audiollm_imda_model_generation(self, input)

            elif self.model_name == "MERaLiON-AudioLLM-Whisper-SEA-LION":
                from model_src.meralion_audiollm_whisper_sea_lion import meralion_audiollm_whisper_sea_lion_model_generation
                return meralion_audiollm_whisper_sea_lion_model_generation(self, input)
            
            elif self.model_name == "gemini-1.5-flash":
                from model_src.gemini_1_5_flash import gemini_1_5_flash_model_generation
                return gemini_1_5_flash_model_generation(self, input)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            elif self.model_name == 'test_temp': return test_temp_model_generation(self, input)

            # MOWE-Audio
            elif self.model_name == "mowe_audio": return mowe_audio_model_generation(self, input)
            elif self.model_name == "multitask-subsetv2:whisper_specaugment+seqcnn8+lora:f3epoch:run2.0:4gpu": return temp_mowe_audio_11_24_model_generation(self, input)
            
            elif self.model_name == "AudioGemma2_v1": return audiogemma_2_v1_model_generation(self, input)
            elif self.model_name == "audiogemma_2_singlish": return audiogemma_2_singlish_model_generation(self, input)
            elif self.model_name == "meralion_audiollm_v1_lora": return meralion_audiollm_v1_lora_model_generation(self, input)
            elif self.model_name == "meralion_audiollm_v1_mse": return meralion_audiollm_v1_mse_model_generation(self, input)
            elif self.model_name == "stage2_whisper3_fft_mlp100_gemma2_9b_lora": return stage2_whisper3_fft_mlp100_gemma2_9b_lora_model_generation(self, input)
            elif self.model_name == "stage2_only_whisper3_fft_mlp100_sealion3_9b_lora": return stage2_only_whisper3_fft_mlp100_sealion3_9b_lora_model_generation(self, input)

            elif self.model_name == 'audiollm_imda': return audiollm_imda_model_generation(self, input)

            elif 'AudioGemma_IMDA_' in self.model_name:
                return audiogemma_imda_model_generation(self, input)

            else:
                raise NotImplementedError("Model {} not implemented yet".format(self.model_name))

