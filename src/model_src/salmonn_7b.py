# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('examples')
import logging
import numpy as np
import torch

from tqdm import trange, tqdm

from SALMONN_7B.model import SALMONN

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "examples/SALMONN_7B/"

def salmonn_7b_model_loader(self):

    # torch.set_default_dtype(torch.float16)

    self.model = SALMONN(
        ckpt         = model_path + "ckpt_path/salmonn_7b_v0.pth",
        whisper_path = model_path + "whisper",
        beats_path   = model_path + "beats_path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        vicuna_path  = model_path + "vicuna",
        low_resource = False
    )
    
    self.model.to(self.device)
    self.model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print('The model has  {:,} parameters'.format(count_parameters(self.model)))


def salmonn_7b_model_generation(self, input):

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

            outputs = self.model.generate(audio_array=chunk, sampling_rate=sampling_rate, prompt=input["instruction"], device=self.device)[0]
            model_predictions.append(outputs)
        
        output = ' '.join(model_predictions)

    # For other tasks, if audio duration is more than 30 seconds, we will take first 30 seconds
    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        audio_array = audio_array[:30 * sampling_rate]
        output = self.model.generate(audio_array=audio_array, sampling_rate=sampling_rate, prompt=input["instruction"], device=self.device)[0]

    # If audio duration is less than 1 second, we will pad the audio to 1 second
    elif audio_duration < 1:
        logger.info('Audio duration is less than 1 second. Padding to 1 second.')
        audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant', constant_values=(0, 0))
        output = self.model.generate(audio_array=audio_array, sampling_rate=sampling_rate, prompt=input["instruction"], device=self.device)[0]

    else:
        output = self.model.generate(audio_array=audio_array, sampling_rate=sampling_rate, prompt=input["instruction"], device=self.device)[0]

    return output


