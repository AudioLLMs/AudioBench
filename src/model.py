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

        self.model_name = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        
        elif self.model_name == "Qwen2-Audio-7B-Instruct":
            from model_src.qwen2_audio_7b_instruct import qwen2_audio_7b_instruct_model_loader
            qwen2_audio_7b_instruct_model_loader(self)

        elif self.model_name == "SALMONN_7B":
            from model_src.salmonn_7b import salmonn_7b_model_loader
            salmonn_7b_model_loader(self)

        elif self.model_name == 'Qwen-Audio-Chat':
            from model_src.qwen_audio_chat import qwen_audio_chat_model_loader
            qwen_audio_chat_model_loader(self)

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
            
            elif self.model_name == "Qwen2-Audio-7B-Instruct":
                from model_src.qwen2_audio_7b_instruct import qwen2_audio_7b_instruct_model_generation
                return qwen2_audio_7b_instruct_model_generation(self, input)

            elif self.model_name == "SALMONN_7B":
                from model_src.salmonn_7b import salmonn_7b_model_generation
                return salmonn_7b_model_generation(self, input)
            
            elif self.model_name == "Qwen-Audio-Chat":
                from model_src.qwen_audio_chat import qwen_audio_chat_model_generation
                return qwen_audio_chat_model_generation(self, input)

            else:
                raise NotImplementedError("Model {} not implemented yet".format(self.model_name))

