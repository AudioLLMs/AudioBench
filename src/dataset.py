import logging
import os

# add parent directory to sys.path
import sys
sys.path.append('.')

from datasets import load_dataset, load_from_disk


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

        if   self.dataset_name == 'cn_college_listen_mcq_test': 
            self.raw_data = load_dataset("AudioLLMs/cn_college_listen_mcq_test")['test']

        elif self.dataset_name == 'slue_p2_sqa5_test': 
            self.raw_data = load_dataset("AudioLLMs/slue_p2_sqa5_test")['test']

        elif self.dataset_name == 'public_sg_speech_qa_test': 
            self.raw_data = load_dataset("AudioLLMs/public_sg_speech_qa_test")['test']

        elif self.dataset_name == 'dream_tts_mcq_test': 
            self.raw_data = load_dataset("AudioLLMs/dream_tts_mcq_test")['test']

        elif self.dataset_name == 'librispeech_test_clean': 
            self.raw_data = load_dataset("AudioLLMs/librispeech_test_clean")['test']

        elif self.dataset_name == 'librispeech_test_other': 
            self.raw_data = load_dataset("AudioLLMs/librispeech_test_other")['test']

        elif self.dataset_name == 'common_voice_15_en_test': 
            self.raw_data = load_dataset("AudioLLMs/common_voice_15_en_test")['test']

        elif self.dataset_name == 'peoples_speech_test': 
            self.raw_data = load_dataset("AudioLLMs/peoples_speech_test")['test']

        elif self.dataset_name == 'gigaspeech_test': 
            self.raw_data = load_dataset("AudioLLMs/gigaspeech_test")['test']

        elif self.dataset_name == 'earnings21_test': 
            self.raw_data = load_dataset("AudioLLMs/earnings21_test")['test']

        elif self.dataset_name == 'earnings22_test': 
            self.raw_data = load_dataset("AudioLLMs/earnings22_test")['test']

        elif self.dataset_name == 'tedlium3_test': 
            self.raw_data = load_dataset("AudioLLMs/tedlium3_test")['test']

        elif self.dataset_name == 'tedlium3_long_form_test': 
            self.raw_data = load_dataset("AudioLLMs/tedlium3_long_form_test")['test']

        elif self.dataset_name == 'openhermes_audio_test': 
            self.raw_data = load_dataset("AudioLLMs/openhermes_instruction_test")['test']

        elif self.dataset_name == 'alpaca_audio_test': 
            self.raw_data = load_dataset("AudioLLMs/alpaca_audio_test")['test']

        elif self.dataset_name == 'audiocaps_test': 
            self.raw_data = load_dataset("AudioLLMs/audiocaps_test")['test']

        elif self.dataset_name == 'wavcaps_test': 
            self.raw_data = load_dataset("AudioLLMs/wavcaps_test")['test']

        elif self.dataset_name == 'clotho_aqa_test': 
            self.raw_data = load_dataset("AudioLLMs/clotho_aqa_test")['test']

        elif self.dataset_name == 'audiocaps_qa_test': 
            self.raw_data = load_dataset("AudioLLMs/audiocaps_qa_test")['test']

        elif self.dataset_name == 'wavcaps_qa_test': 
            self.raw_data = load_dataset("AudioLLMs/wavcaps_qa_test")['test']

        elif self.dataset_name == 'voxceleb_accent_test': 
            self.raw_data = load_dataset("AudioLLMs/voxceleb_accent_test")['test']

        elif self.dataset_name == 'voxceleb_gender_test': 
            self.raw_data = load_dataset("AudioLLMs/voxceleb_gender_test")['test']

        elif self.dataset_name == 'iemocap_gender_test': 
            self.raw_data = load_dataset("AudioLLMs/iemocap_gender_recognition")['test']

        elif self.dataset_name == 'iemocap_emotion_test': 
            self.raw_data  = load_dataset("AudioLLMs/iemocap_emotion_recognition")['test']

        elif self.dataset_name == 'meld_sentiment_test': 
            self.raw_data = load_dataset("AudioLLMs/meld_sentiment_test")['test']

        elif self.dataset_name == 'meld_emotion_test': 
            self.raw_data = load_dataset("AudioLLMs/meld_emotion_test")['test']

        elif self.dataset_name == 'covost2_en_id_test': 
            self.raw_data = load_dataset("AudioLLMs/covost2_en_id_test")['test']

        elif self.dataset_name == 'covost2_en_zh_test': 
            self.raw_data = load_dataset("AudioLLMs/covost2_en_zh_test")['test']

        elif self.dataset_name == 'covost2_en_ta_test': 
            self.raw_data = load_dataset("AudioLLMs/covost2_en_ta_test")['test']

        elif self.dataset_name == 'covost2_id_en_test': 
            self.raw_data = load_dataset("AudioLLMs/covost2_id_en_test")['test']

        elif self.dataset_name == 'covost2_zh_en_test': 
            self.raw_data = load_dataset("AudioLLMs/covost2_zh_en_test")['test']

        elif self.dataset_name == 'covost2_ta_en_test': 
            self.raw_data = load_dataset("AudioLLMs/covost2_ta_en_test")['test']

        elif self.dataset_name == 'aishell_asr_zh_test': 
            self.raw_data = load_dataset("AudioLLMs/aishell_1_zh_test")['test']

        elif self.dataset_name == 'spoken_squad_test': 
            self.raw_data = load_dataset("AudioLLMs/spoken_squad_test")['test']

        elif self.dataset_name == 'muchomusic_test': 
            self.raw_data = load_dataset("AudioLLMs/mu_chomusic_test")['test']

        elif self.dataset_name == 'imda_part1_asr_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='ASR-PART1-Test')['train']

        elif self.dataset_name == 'imda_part2_asr_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='ASR-PART2-Test')['train']

        elif self.dataset_name == 'imda_part3_30s_asr_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='ASR-PART3-Test')['train']

        elif self.dataset_name == 'imda_part4_30s_asr_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='ASR-PART4-Test')['train']

        elif self.dataset_name == 'imda_part5_30s_asr_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='ASR-PART5-Test')['train']

        elif self.dataset_name == 'imda_part6_30s_asr_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='ASR-PART6-Test')['train']

        elif self.dataset_name == 'imda_part3_30s_sqa_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SQA-PART3-Test')['train']

        elif self.dataset_name == 'imda_part4_30s_sqa_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SQA-PART4-Test')['train']

        elif self.dataset_name == 'imda_part5_30s_sqa_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SQA-PART5-Test')['train']

        elif self.dataset_name == 'imda_part6_30s_sqa_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SQA-PART6-Test')['train']

        elif self.dataset_name == 'imda_part3_30s_ds_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SDS-PART3-Test')['train']

        elif self.dataset_name == 'imda_part4_30s_ds_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SDS-PART4-Test')['train']

        elif self.dataset_name == 'imda_part5_30s_ds_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SDS-PART5-Test')['train']

        elif self.dataset_name == 'imda_part6_30s_ds_human_test': 
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='SDS-PART6-Test')['train']

        elif self.dataset_name == 'imda_ar_sentence':
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='PQA-AR-Sentence-Test')['train']

        elif self.dataset_name == 'imda_ar_dialogue':
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='PQA-AR-Dialogue-Test')['train']

        elif self.dataset_name == 'imda_gr_sentence':
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='PQA-GR-Sentence-Test')['train']

        elif self.dataset_name == 'imda_gr_dialogue':
            self.raw_data = load_dataset('MERaLiON/Multitask-National-Speech-Corpus-v1', data_dir='PQA-GR-Dialogue-Test')['train']

        elif self.dataset_name == 'seame_dev_man':
            self.raw_data = load_dataset("AudioLLMs/seame_dev_man")['test']

        elif self.dataset_name == 'seame_dev_sge':
            self.raw_data = load_dataset("AudioLLMs/seame_dev_sge")['test']

        elif self.dataset_name == 'mmau_mini':
            self.raw_data = load_dataset("AudioLLMs/MMAU-mini")['test']

        elif self.dataset_name == 'gigaspeech2_thai':
            self.raw_data = load_dataset("AudioLLMs/gigaspeech2-test", data_dir='th-test')['train']

        elif self.dataset_name == 'gigaspeech2_indo':
            self.raw_data = load_dataset("AudioLLMs/gigaspeech2-test", data_dir='id-test')['train']

        elif self.dataset_name == 'gigaspeech2_viet':
            self.raw_data = load_dataset("AudioLLMs/gigaspeech2-test", data_dir='vi-test')['train']

        # Private
        elif self.dataset_name == 'ytb_asr_batch1':
            self.raw_data = load_from_disk("data/3_private_data/ytb_asr_batch1")

        elif self.dataset_name == 'ytb_asr_batch2':
            self.raw_data = load_from_disk("data/3_private_data/ytb_asr_batch2")

        elif self.dataset_name == 'ytb_sqa_batch1':
            self.raw_data = load_from_disk("data/3_private_data/ytb_sqa_batch1")

        elif self.dataset_name == 'ytb_sds_batch1':
            self.raw_data = load_from_disk("data/3_private_data/ytb_sds_batch1")

        elif self.dataset_name == 'ytb_pqa_batch1':
            self.raw_data = load_from_disk("data/3_private_data/ytb_pqa_batch1")
            
        elif self.dataset_name == 'cna_test': 
            self.raw_data = load_from_disk("data/3_private_data/cna_ASR_v3")

        elif self.dataset_name == 'idpc_test': 
            self.raw_data = load_from_disk("data/3_private_data/idpc_long_ASR_v1")

        elif self.dataset_name == 'parliament_test': 
            self.raw_data = load_from_disk("data/3_private_data/parliament_long_ASR_v1")

        elif self.dataset_name == 'ukusnews_test': 
            self.raw_data = load_from_disk("data/3_private_data/ukusnews_long_ASR_v1")

        elif self.dataset_name == 'mediacorp_test': 
            self.raw_data = load_from_disk("data/3_private_data/mediacorp_long_ASR_v1")

        elif self.dataset_name == 'idpc_short_test': 
            self.raw_data = load_from_disk("data/3_private_data/idpc_short_ASR_v1")

        elif self.dataset_name == 'parliament_short_test': 
            self.raw_data = load_from_disk("data/3_private_data/parliament_short_ASR_v1")

        elif self.dataset_name == 'ukusnews_short_test': 
            self.raw_data = load_from_disk("data/3_private_data/ukusnews_short_ASR_v1")

        elif self.dataset_name == 'mediacorp_short_test': 
            self.raw_data = load_from_disk("data/3_private_data/mediacorp_short_ASR_v1")
        
        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
        logger.info("= = "*20)


    def data_format(self):

        # if samples less than requested samples
        if len(self.raw_data) < self.number_of_samples:
            self.number_of_samples = len(self.raw_data)
            logger.info("Number of samples requested is more than available samples. Setting number of samples to {}".format(self.number_of_samples))

        if self.dataset_name == 'cn_college_listen_mcq_test': 
            from dataset_src.cn_college_listen_mcq_test import cn_college_listen_mcq_test_dataset
            self.dataset_processor = cn_college_listen_mcq_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'slue_p2_sqa5_test':
            from dataset_src.slue_p2_sqa5_test import slue_p2_sqa5_test_dataset
            self.dataset_processor = slue_p2_sqa5_test_dataset(self.raw_data, self.number_of_samples)
            
        elif self.dataset_name == 'public_sg_speech_qa_test': 
            from dataset_src.public_sg_speech_qa_test import public_sg_speech_qa_test_dataset
            self.dataset_processor   = public_sg_speech_qa_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'dream_tts_mcq_test': 
            from dataset_src.dream_tts_mcq_test import dream_tts_mcq_test_dataset
            self.dataset_processor = dream_tts_mcq_test_dataset(self.raw_data, self.number_of_samples)
        
        elif self.dataset_name == 'librispeech_test_clean': 
            from dataset_src.librispeech_test_clean import librispeech_test_clean_dataset
            self.dataset_processor = librispeech_test_clean_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'librispeech_test_other': 
            from dataset_src.librispeech_test_other import librispeech_test_other_dataset
            self.dataset_processor = librispeech_test_other_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'common_voice_15_en_test':
            from dataset_src.common_voice_15_en_test import common_voice_15_en_test_dataset
            self.dataset_processor = common_voice_15_en_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'peoples_speech_test': 
            from dataset_src.peoples_speech_test import peoples_speech_test_dataset
            self.dataset_processor = peoples_speech_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'gigaspeech_test': 
            from dataset_src.gigaspeech_test import gigaspeech_test_dataset
            self.dataset_processor = gigaspeech_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'earnings21_test': 
            from dataset_src.earnings21_test import earnings21_test_dataset
            self.dataset_processor = earnings21_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'earnings22_test': 
            from dataset_src.earnings22_test import earnings22_test_dataset
            self.dataset_processor = earnings22_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'tedlium3_test': 
            from dataset_src.tedlium3_test import tedlium3_test_dataset
            self.dataset_processor = tedlium3_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'tedlium3_long_form_test': 
            from dataset_src.tedlium3_long_form_test import tedlium3_long_form_test_dataset
            self.dataset_processor = tedlium3_long_form_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'openhermes_audio_test': 
            from dataset_src.openhermes_audio_test import openhermes_audio_test_dataset
            self.dataset_processor = openhermes_audio_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'alpaca_audio_test': 
            from dataset_src.alpaca_audio_test import alpaca_audio_test_dataset
            self.dataset_processor = alpaca_audio_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'audiocaps_test': 
            from dataset_src.audiocaps_test import audiocaps_test_dataset
            self.dataset_processor = audiocaps_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'wavcaps_test': 
            from dataset_src.wavcaps_test import wavcaps_test_dataset
            self.dataset_processor = wavcaps_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'clotho_aqa_test': 
            from dataset_src.clotho_aqa_test import clotho_aqa_test_dataset
            self.dataset_processor = clotho_aqa_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'audiocaps_qa_test': 
            from dataset_src.audiocaps_qa_test import audiocaps_qa_test_dataset
            self.dataset_processor = audiocaps_qa_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'wavcaps_qa_test': 
            from dataset_src.wavcaps_qa_test import wavcaps_qa_test_dataset
            self.dataset_processor = wavcaps_qa_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'voxceleb_accent_test': 
            from dataset_src.voxceleb_accent_test import voxceleb_accent_test_dataset
            self.dataset_processor = voxceleb_accent_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'voxceleb_gender_test': 
            from dataset_src.voxceleb_gender_test import voxceleb_gender_test_dataset
            self.dataset_processor = voxceleb_gender_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'iemocap_gender_test':
            from dataset_src.iemocap_gender_test import iemocap_gender_test_dataset 
            self.dataset_processor = iemocap_gender_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'iemocap_emotion_test': 
            from dataset_src.iemocap_emotion_test import iemocap_emotion_test_dataset
            self.dataset_processor = iemocap_emotion_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'meld_sentiment_test':
            from dataset_src.meld_sentiment_test import meld_sentiment_test_dataset
            self.dataset_processor = meld_sentiment_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'meld_emotion_test': 
            from dataset_src.meld_emotion_test import meld_emotion_test_dataset
            self.dataset_processor = meld_emotion_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_en_id_test':
            from dataset_src.covost2_en_id_test import covost2_en_id_test_dataset
            self.dataset_processor = covost2_en_id_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_en_zh_test': 
            from dataset_src.covost2_en_zh_test import covost2_en_zh_test_dataset
            self.dataset_processor = covost2_en_zh_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_en_ta_test': 
            from dataset_src.covost2_en_ta_test import covost2_en_ta_test_dataset
            self.dataset_processor = covost2_en_ta_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_id_en_test': 
            from dataset_src.covost2_id_en_test import covost2_id_en_test_dataset
            self.dataset_processor = covost2_id_en_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_zh_en_test': 
            from dataset_src.covost2_zh_en_test import covost2_zh_en_test_dataset
            self.dataset_processor = covost2_zh_en_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'covost2_ta_en_test': 
            from dataset_src.covost2_ta_en_test import covost2_ta_en_test_dataset
            self.dataset_processor = covost2_ta_en_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'aishell_asr_zh_test': 
            from dataset_src.aishell_asr_zh_test import aishell_asr_zh_test_dataset
            self.dataset_processor = aishell_asr_zh_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'spoken_squad_test': 
            from dataset_src.spoken_squad_test import spoken_squad_test_dataset
            self.dataset_processor = spoken_squad_test_dataset(self.raw_data, self.number_of_samples)
        
        elif self.dataset_name == 'muchomusic_test': 
            from dataset_src.mu_chomusic_test import mu_chomusic_test_dataset
            self.dataset_processor = mu_chomusic_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part1_asr_test': 
            from dataset_src.imda_part1_asr_test import imda_part1_asr_test_dataset
            self.dataset_processor = imda_part1_asr_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part2_asr_test':
            from dataset_src.imda_part2_asr_test import imda_part2_asr_test_dataset
            self.dataset_processor = imda_part2_asr_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part3_30s_asr_test':
            from dataset_src.imda_part3_30s_asr_test import imda_part3_30s_asr_test_dataset
            self.dataset_processor = imda_part3_30s_asr_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part4_30s_asr_test':
            from dataset_src.imda_part4_30s_asr_test import imda_part4_30s_asr_test_dataset
            self.dataset_processor = imda_part4_30s_asr_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part5_30s_asr_test': 
            from dataset_src.imda_part5_30s_asr_test import imda_part5_30s_asr_test_dataset
            self.dataset_processor = imda_part5_30s_asr_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part6_30s_asr_test': 
            from dataset_src.imda_part6_30s_asr_test import imda_part6_30s_asr_test_dataset
            self.dataset_processor = imda_part6_30s_asr_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part3_30s_sqa_human_test': 
            from dataset_src.imda_part3_30s_sqa_human_test import imda_part3_30s_sqa_human_test_dataset
            self.dataset_processor = imda_part3_30s_sqa_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part4_30s_sqa_human_test': 
            from dataset_src.imda_part4_30s_sqa_human_test import imda_part4_30s_sqa_human_test_dataset
            self.dataset_processor = imda_part4_30s_sqa_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part5_30s_sqa_human_test': 
            from dataset_src.imda_part5_30s_sqa_human_test import imda_part5_30s_sqa_human_test_dataset
            self.dataset_processor = imda_part5_30s_sqa_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part6_30s_sqa_human_test': 
            from dataset_src.imda_part6_30s_sqa_human_test import imda_part6_30s_sqa_human_test_dataset
            self.dataset_processor = imda_part6_30s_sqa_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part3_30s_ds_human_test': 
            from dataset_src.imda_part3_30s_ds_human_test import imda_part3_30s_ds_human_test_dataset
            self.dataset_processor = imda_part3_30s_ds_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part4_30s_ds_human_test': 
            from dataset_src.imda_part4_30s_ds_human_test import imda_part4_30s_ds_human_test_dataset
            self.dataset_processor = imda_part4_30s_ds_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part5_30s_ds_human_test': 
            from dataset_src.imda_part5_30s_ds_human_test import imda_part5_30s_ds_human_test_dataset
            self.dataset_processor = imda_part5_30s_ds_human_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_part6_30s_ds_human_test': 
            from dataset_src.imda_part6_30s_ds_human_test import imda_part6_30s_ds_human_test_dataset
            self.dataset_processor = imda_part6_30s_ds_human_test_dataset(self.raw_data, self.number_of_samples)
        
        elif self.dataset_name == 'imda_ar_sentence':
            from dataset_src.imda_ar_sentence import imda_ar_sentence_test_dataset
            self.dataset_processor = imda_ar_sentence_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_ar_dialogue':
            from dataset_src.imda_ar_dialogue import imda_ar_dialogue_test_dataset
            self.dataset_processor = imda_ar_dialogue_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_gr_sentence':
            from dataset_src.imda_gr_sentence import imda_gr_sentence_test_dataset
            self.dataset_processor = imda_gr_sentence_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'imda_gr_dialogue':
            from dataset_src.imda_gr_dialogue import imda_gr_dialogue_test_dataset
            self.dataset_processor = imda_gr_dialogue_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'mmau_mini':
            from dataset_src.mmau_mini import mmau_mini_test_dataset
            self.dataset_processor = mmau_mini_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'gigaspeech2_thai':
            from dataset_src.gigaspeech2_thai import gigaspeech2_thai_test_dataset
            self.dataset_processor = gigaspeech2_thai_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'gigaspeech2_indo':
            from dataset_src.gigaspeech2_indo import gigaspeech2_indo_test_dataset
            self.dataset_processor = gigaspeech2_indo_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'gigaspeech2_viet':
            from dataset_src.gigaspeech2_viet import gigaspeech2_viet_test_dataset
            self.dataset_processor = gigaspeech2_viet_test_dataset(self.raw_data, self.number_of_samples)


        # Private
        elif self.dataset_name == 'ytb_asr_batch1':
            from dataset_src.ytb_asr_batch1 import ytb_asr_batch1_dataset
            self.dataset_processor = ytb_asr_batch1_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'ytb_asr_batch2':
            from dataset_src.ytb_asr_batch2 import ytb_asr_batch2_dataset
            self.dataset_processor = ytb_asr_batch2_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'ytb_sqa_batch1':
            from dataset_src.ytb_sqa_batch1 import ytb_sqa_batch1_dataset
            self.dataset_processor = ytb_sqa_batch1_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'ytb_sds_batch1':
            from dataset_src.ytb_sds_batch1 import ytb_sds_batch1_dataset
            self.dataset_processor = ytb_sds_batch1_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'ytb_pqa_batch1':
            from dataset_src.ytb_pqa_batch1 import ytb_pqa_batch1_dataset
            self.dataset_processor = ytb_pqa_batch1_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'seame_dev_man':
            from dataset_src.seame_dev_man import seame_dev_man_dataset
            self.dataset_processor = seame_dev_man_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'seame_dev_sge':
            from dataset_src.seame_dev_sge import seame_dev_sge_dataset
            self.dataset_processor = seame_dev_sge_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'cna_test': 
            from dataset_src.cna_test import cna_test_dataset
            self.dataset_processor = cna_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'idpc_test': 
            from dataset_src.idpc_test import idpc_test_dataset
            self.dataset_processor = idpc_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'parliament_test': 
            from dataset_src.parliament_test import parliament_test_dataset
            self.dataset_processor = parliament_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'ukusnews_test': 
            from dataset_src.ukusnews_test import ukusnews_test_dataset
            self.dataset_processor = ukusnews_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'mediacorp_test': 
            from dataset_src.mediacorp_test import mediacorp_test_dataset
            self.dataset_processor = mediacorp_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'idpc_short_test': 
            from dataset_src.idpc_short_test import idpc_short_test_dataset
            self.dataset_processor = idpc_short_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'parliament_short_test':
            from dataset_src.parliament_short_test import parliament_short_test_dataset
            self.dataset_processor = parliament_short_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'ukusnews_short_test':
            from dataset_src.ukusnews_short_test import ukusnews_short_test_dataset
            self.dataset_processor = ukusnews_short_test_dataset(self.raw_data, self.number_of_samples)

        elif self.dataset_name == 'mediacorp_short_test': 
            from dataset_src.mediacorp_short_test import mediacorp_short_test_dataset
            self.dataset_processor = mediacorp_short_test_dataset(self.raw_data, self.number_of_samples)

        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

        self.input_data = self.dataset_processor.prepare_model_input()

