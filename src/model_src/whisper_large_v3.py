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

whisper_model_path = "openai/whisper-large-v3"


def whisper_large_v3_model_loader(self):

    self.whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map="auto")
    self.whisper_processor = AutoProcessor.from_pretrained(whisper_model_path)
    self.whisper_pipe      = pipeline(
                    "automatic-speech-recognition",
                    model              = self.whisper_model,
                    tokenizer          = self.whisper_processor.tokenizer,
                    feature_extractor  = self.whisper_processor.feature_extractor,
                    max_new_tokens     = 128,
                    chunk_length_s     = 30,
                    batch_size         = 16,
                    return_timestamps  = False,
                    torch_dtype        = torch.float16,
                    device_map         = "auto",
                )
    self.whisper_model.eval()

    logging.info(f"Model loaded from {whisper_model_path}.")


def whisper_large_v3_model_generation(self, sample):

    if sample['task_type'] == 'ASR':
        #whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "en"})['text'].strip()
        whisper_output = self.whisper_pipe(sample['audio'])['text'].strip()
        return whisper_output

    elif sample['task_type'] == "ASR-ZH":
        whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "zh"})['text'].strip()
        return whisper_output
    
    elif sample['task_type'] in ["ST-ID-EN",
                                 "ST-TA-EN",
                                 "ST-ZH-EN",
                                 ]:
        whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"task": "translate", "language": "en"})['text'].strip()
        return whisper_output
    
    else:
        raise NotImplementedError(f"Whisper does not support other task: {sample['task_type']}.")

    return whisper_output

    