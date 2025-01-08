


import torch

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
import librosa

from tqdm import tqdm
import logging
import examples.AudioGemma2.modules_audio_gemma2.inference as inference

from hydra import initialize, compose



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cp_path = "examples/AudioGemma2/checkpoints/gemma2_2b_datasets_0908_0916/model.safetensors"
config_path="../../examples/AudioGemma2/checkpoints/gemma2_2b_datasets_0908_0916"
config_name="training_arguments"

with initialize(config_path=config_path):
    cfg = compose(config_name=config_name)

def load_model(_cfg):
    
    model = inference.load_model(_cfg, cp_path)

    data_collator = inference.SpeechTextDataCollator(
        feature_extractor=model.speech_encoder.feature_extractor,
        llm_tokenizer=model.text_decoder.tokenizer,
        use_audio_encoder=False,
        is_train=False,
        asr_normalize_text=cfg.asr_normalize_text,
        use_system_prompt=cfg.model.text_decoder.use_system_prompt,
    )
    return model, data_collator

def audiogemma_2_v1_model_loader(self):

    print("model loading", flush=True)
    model, data_collator = load_model(cfg)
    print("model loaded", flush=True)

    self.model = model
    self.model_arguments = cfg
    self.data_collator = data_collator



@dataclass
class DataCollator:
    speech_processor: Any
    llm_tokenizer: Any
    use_audio_encoder: bool
    is_train: bool = True
    asr_normalize_text: bool = False

    def convert_to_chat_template(self, instruction):

        conversation = [
            {
                "role": "system",
                "content": "You are a speech-text multimodal assistant. You would be given some speech tokens and text prompt tokens. Your job is to answer the question in the text prompt based on the context in speech tokens.",
            },
            {
                "role": "user",
                "content": f"Given the following speech tokens: \n<speech><SpeechHere></speech>\n\n{instruction}",
            },
        ]

        instruction_in_llm_template = self.llm_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False if self.is_train else True,
        )

        return instruction_in_llm_template

    def split_text_prompts(self, instructions):

        prompts_left = []
        prompts_right = []

        for instruction in instructions:
            instruction_in_llm_template = self.convert_to_chat_template(
                instruction=instruction
            )

            prompt_left, prompt_right = instruction_in_llm_template.split(
                "<SpeechHere>")

            prompts_left.append(prompt_left)
            prompts_right.append(prompt_right)

        return prompts_left, prompts_right

    def tokenize_left_prompts(self, prompts):
        return self.llm_tokenizer(
            text=prompts,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
            padding=True,
            truncation="longest_first",
            max_length=256,
        )

    def tokenize_right_prompts(self, prompts):
        if ("zephyr-7b-beta" in self.llm_tokenizer.name_or_path) or ("Phi" in self.llm_tokenizer.name_or_path):
            add_special_tokens = True
        else:
            add_special_tokens = False

        self.llm_tokenizer.add_bos_token = False
        prompts = self.llm_tokenizer(
            text                  = prompts,
            return_tensors        = "pt",
            add_special_tokens    = add_special_tokens,
            return_attention_mask = True,
            return_token_type_ids = True,
            padding               = True,
            truncation            = "longest_first",
            max_length            = 256,
        )
        self.llm_tokenizer.add_bos_token = True
        return prompts

    def __call__(self, examples: List) -> Dict[str, torch.Tensor]:

        speech_contexts = [example["audio"] for example in examples]
        speech_contexts_features = {
            processor_name: speech_processor(
            [resample_fn(s) for s in speech_contexts],
            return_tensors="pt", return_attention_mask=True, **processor_args
        ) for processor_name, (speech_processor, processor_args, resample_fn) in self.speech_processor.items()
        }

        instructions = [example["instruction"] for example in examples]

        text_prompts_left, text_prompts_right = self.split_text_prompts(
            instructions)

        text_prompts_left = self.tokenize_left_prompts(text_prompts_left)
        text_prompts_right = self.tokenize_right_prompts(text_prompts_right)

        return {
            "speech_contexts_features": speech_contexts_features,
            "text_prompts_left": text_prompts_left,
            "text_prompts_right": text_prompts_right,
            "text_prompts_right_wo_responses": text_prompts_right,  # not used in model
            "audio_contexts_features": torch.zeros(5),
            "audio_contexts_mask": torch.zeros(5),
        }



def audiogemma_2_v1_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. For ASR task, we will chunk and infer separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):
            
            # if chunk is less than 1 second, pad it to 1 second
            if len(chunk) < sampling_rate:
                chunk = np.pad(chunk, (0, sampling_rate - len(chunk)), 'constant', constant_values=(0, 0))
            
            audio_array = chunk

            samples = [{
                'audio'      : audio_array,
                'instruction': input["text"],
            }]
            batch = self.data_collator(samples)

            batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
                else v for k, v in batch.items()}
            for k, v in batch["speech_contexts_features"].items():
                v = v.to("cuda")
                
            with torch.no_grad():
                output = self.model.generate(**batch, synced_gpu=False)[0]

            model_predictions.append(output)
        
        output = ' '.join(model_predictions)

    else:
        
        # If audio duration is less than 1 second, we will pad the audio to 1 second
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant', constant_values=(0, 0))

        # For other tasks, if audio duration is more than 30 seconds, we will take first 30 seconds
        elif audio_duration > 30:
            logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
            audio_array = audio_array[:30 * sampling_rate]
            
        samples = [{
            'audio'      : audio_array,
            'instruction': input["text"],
        }]
        batch = self.data_collator(samples)
        batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
            else v for k, v in batch.items()}
        for k, v in batch["speech_contexts_features"].items():
            v = v.to("cuda")

        with torch.no_grad():
            output = self.model.generate(**batch, synced_gpu=False)[0]

    return output






def main(samples: List[Dict[str, np.ndarray|str]]) -> List[str]:

    model, model_arguments = load_model()

    data_collator = DataCollator(
        speech_processor   = model.get_speech_processor(),
        llm_tokenizer      = model.get_llm_tokenizer(),
        use_audio_encoder  = False,
        is_train           = False,
        asr_normalize_text = model_arguments.asr_normalize_text,
    )

    batch = data_collator(samples)
    batch = {k: v.to(device) if k not in ["text_labels", "task_names", "speech_contexts_features"]
        else v for k, v in batch.items()}
    for k, v in batch["speech_contexts_features"].items():
        v = v.to("cuda")

    with torch.no_grad():
        predictions = model.generate(**batch, synced_gpu=False)

    print(predictions)

    return predictions


if __name__ == "__main__":

    samples=[{
        "audio"      : librosa.load('tmp.wav', sr=16000)[0],
        "instruction": "Please follow the instruction in the speech"
         }]

    main(samples)
