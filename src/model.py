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
            from model_src.whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_loader
            whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_loader(self)
        
        elif self.model_name == "Qwen2-Audio-7B-Instruct":
            from model_src.qwen2_audio_7b_instruct import qwen2_audio_7b_instruct_model_loader
            qwen2_audio_7b_instruct_model_loader(self)

        elif self.model_name == "SALMONN_7B":
            from model_src.salmonn_7b import salmonn_7b_model_loader
            salmonn_7b_model_loader(self)

        elif self.model_name == 'WavLLM_fairseq': 
            from model_src.wavllm_fairseq import wavllm_fairseq_model_loader
            wavllm_fairseq_model_loader(self)

        elif self.model_name == 'Qwen-Audio-Chat':
            from model_src.qwen_audio_chat import qwen_audio_chat_model_loader
            qwen_audio_chat_model_loader(self)

        elif self.model_name == 'MERaLiON-AudioLLM-Whisper-SEA-LION':
            from model_src.meralion_audiollm_whisper_sea_lion import meralion_audiollm_whisper_sea_lion_model_loader
            meralion_audiollm_whisper_sea_lion_model_loader(self)

        elif self.model_name == 'gemini-1.5-flash':
            from model_src.gemini_1_5_flash import gemini_1_5_flash_model_loader
            gemini_1_5_flash_model_loader(self)

        elif self.model_name == 'gemini-2-flash':
            from model_src.gemini_2_flash import gemini_2_flash_model_loader
            gemini_2_flash_model_loader(self)

        elif self.model_name == 'whisper_large_v3':
            from model_src.whisper_large_v3 import whisper_large_v3_model_loader
            whisper_large_v3_model_loader(self)

        elif self.model_name == 'whisper_large_v2':
            from model_src.whisper_large_v2 import whisper_large_v2_model_loader
            whisper_large_v2_model_loader(self)

        elif self.model_name == 'gpt-4o-audio':
            from model_src.gpt_4o_audio import gpt_4o_audio_model_loader
            gpt_4o_audio_model_loader(self)

        elif self.model_name == 'phi_4_multimodal_instruct':
            from model_src.phi_4_multimodal_instruct import phi_4_multimodal_instruct_model_loader
            phi_4_multimodal_instruct_model_loader(self)

        elif self.model_name == 'seallms_audio_7b':
            from model_src.seallms_audio_7b import seallms_audio_7b_model_loader
            seallms_audio_7b_model_loader(self)

        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))


    def generate(self, input):

        with torch.no_grad():
            if self.model_name == "cascade_whisper_large_v3_llama_3_8b_instruct": 
                from model_src.whisper_large_v3_with_llama_3_8b_instruct import whisper_large_v3_with_llama_3_8b_instruct_model_generation
                return whisper_large_v3_with_llama_3_8b_instruct_model_generation(self, input)
            
            elif self.model_name == "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct":
                from model_src.whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct import whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_generation
                return whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_generation(self, input)
            
            elif self.model_name == "Qwen2-Audio-7B-Instruct":
                from model_src.qwen2_audio_7b_instruct import qwen2_audio_7b_instruct_model_generation
                return qwen2_audio_7b_instruct_model_generation(self, input)

            elif self.model_name == "SALMONN_7B":
                from model_src.salmonn_7b import salmonn_7b_model_generation
                return salmonn_7b_model_generation(self, input)
            
            elif self.model_name == "WavLLM_fairseq":
                from model_src.wavllm_fairseq import wavllm_fairseq_model_generation
                return wavllm_fairseq_model_generation(self, input)
            
            elif self.model_name == "Qwen-Audio-Chat":
                from model_src.qwen_audio_chat import qwen_audio_chat_model_generation
                return qwen_audio_chat_model_generation(self, input)
            
            elif self.model_name == "MERaLiON-AudioLLM-Whisper-SEA-LION":
                from model_src.meralion_audiollm_whisper_sea_lion import meralion_audiollm_whisper_sea_lion_model_generation
                return meralion_audiollm_whisper_sea_lion_model_generation(self, input)
            
            elif self.model_name == "gemini-1.5-flash":
                from model_src.gemini_1_5_flash import gemini_1_5_flash_model_generation
                return gemini_1_5_flash_model_generation(self, input)

            elif self.model_name == "gemini-2-flash":
                from model_src.gemini_2_flash import gemini_2_flash_model_generation
                return gemini_2_flash_model_generation(self, input)

            elif self.model_name == "whisper_large_v3":
                from model_src.whisper_large_v3 import whisper_large_v3_model_generation
                return whisper_large_v3_model_generation(self, input)

            elif self.model_name == "whisper_large_v2":
                from model_src.whisper_large_v2 import whisper_large_v2_model_generation
                return whisper_large_v2_model_generation(self, input)

            elif self.model_name == "gpt-4o-audio":
                from model_src.gpt_4o_audio import gpt_4o_audio_model_generation
                return gpt_4o_audio_model_generation(self, input)

            elif self.model_name == 'phi_4_multimodal_instruct':
                from model_src.phi_4_multimodal_instruct import phi_4_multimodal_instruct_model_generation
                return phi_4_multimodal_instruct_model_generation(self, input)

            elif self.model_name == 'seallms_audio_7b':
                from model_src.seallms_audio_7b import seallms_audio_7b_model_generation
                return seallms_audio_7b_model_generation(self, input)

            else:
                raise NotImplementedError("Model {} not implemented yet".format(self.model_name))

