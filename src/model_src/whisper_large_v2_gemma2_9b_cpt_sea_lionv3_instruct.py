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

whisper_model_path = "openai/whisper-large-v2"
llm_model_path = "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct"

def whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_loader(self):

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

    self.llm_tokenizer           = AutoTokenizer.from_pretrained(llm_model_path, padding_side='left')
    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
    self.llm_model               = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map="auto", torch_dtype=torch.float16)
    self.llm_model.eval()

    logging.info(f"Model loaded from {whisper_model_path} and {llm_model_path}.")


def whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct_model_generation(self, sample):

    if sample['task_type'] == 'ASR':
        whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "en"})['text'].strip()
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
        whisper_output = self.whisper_pipe(sample['audio'], generate_kwargs={"language": "en"})['text'].strip()

        instruction = sample['instruction']

        PROMPT_TEMPLATE = """\
            [Audio Transcriptions]
            {whisper_output}

            [Question]
            {instruction}

            [System]
            Please answer the question based on the audio transcription provided above. 
            Ensure that your response adheres to the following format:
            
            Answer: (Provide a precise and concise answer here.)
            """
        
        batch_input = [PROMPT_TEMPLATE.format(whisper_output=whisper_output, instruction=instruction)]

        # If speech instruction task, then only use whisper_output
        if sample['task_type'] == "SI":
            batch_input = [whisper_output]

        batch_input_templated = []
        for sample in batch_input:    
            messages = [
                {"role": "user", "content": sample},
            ]
            sample_templated = self.llm_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
            batch_input_templated.append(sample_templated)

        batch_input = batch_input_templated

        encoded_batch        = self.llm_tokenizer(batch_input, return_tensors="pt", padding=True).to(self.llm_model.device)
        generated_ids        = self.llm_model.generate(**encoded_batch, max_new_tokens=500, pad_token_id=self.llm_tokenizer.eos_token_id)
        generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
        decoded_batch_output = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if 'Answer: ' in decoded_batch_output:
            decoded_batch_output = decoded_batch_output.split('Answer: ')[1].strip()
        
        return decoded_batch_output

