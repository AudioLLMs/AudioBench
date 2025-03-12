import os

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging

import time

import subprocess
import soundfile as sf


# Install fairseq 'pip install --editable ./'


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

root_path = "/project"


def wavllm_fairseq_model_loader(self):
    logger.info("For WavLLM, the model is executed in the fairseq framework. Therefore, the model is loaded directly in the inference function.")


def wavllm_fairseq_model_generation(self, all_input_data):
    # For WavLLM model, as it is executed in the fairseq framework
    # It is not possible to use the transformers pipeline for generation
    # Instead, the model is loaded and executed directly

    dataset_name = self.dataset_name

    os.makedirs(f'tmp/wavllm/{dataset_name}', exist_ok=True)


    chunk_index         = 0
    sample_chunk_number = []

    # Initialize the pointer file
    with open(f'tmp/wavllm/{dataset_name}/wavllm_test.tsv', 'w') as f:
        f.write(f"id\taudio\tn_frames\tprompt\ttgt_text\twith_speech\n")
        f.write(f"{chunk_index}\t{root_path}/AudioBench/src/model_src/wavllm_fairseq_cold_start.wav\t0\tPlease transcribe.\tPlacehoder.\tTrue\n")
        chunk_index += 1

    for input_data in all_input_data:

        audio_array   = input_data["audio"]["array"]
        sampling_rate = input_data["audio"]["sampling_rate"]

        length_of_audio = len(audio_array) / sampling_rate

        if input_data['task_type'] == 'ASR' and length_of_audio > 100:
            logger.info("Audio is longer than 100 seconds. Chunking the audio.")
            
            audio_chunks = []
            for i in range(0, len(audio_array), 100 * sampling_rate):
                audio_chunks.append(audio_array[i:i + 100 * sampling_rate])

            for audio_chunk in audio_chunks:
                instruction   = input_data["instruction"].replace('\n', ' ').strip() # Wavllm does not support newline characters for instructions
                answer        = "==== No reference Answer ===="
                audio_path    = f"tmp/wavllm/{dataset_name}/audios/audio_{chunk_index}.wav"

                # Write the audio to a file
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                sf.write(audio_path, audio_chunk, sampling_rate)

                # Write the metadata to the pointer file
                with open(f'tmp/wavllm/{dataset_name}/wavllm_test.tsv', 'a') as f:
                    f.write(f"{chunk_index}\t{root_path}/AudioBench/{audio_path}\t0\t{instruction}\t{answer}\tTrue\n")
                chunk_index += 1

            sample_chunk_number.append(len(audio_chunks))

        else:

            if length_of_audio > 100:
                logger.info("Audio is longer than 100 seconds. Truncating to 100 seconds.")
                audio_array = audio_array[:sampling_rate*100]

            instruction   = input_data["instruction"].replace('\n', ' ').strip()
            answer        = "==== No reference Answer ===="
            audio_path    = f"tmp/wavllm/{dataset_name}/audios/audio_{chunk_index}.wav"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            sf.write(audio_path, audio_array, sampling_rate)

            with open(f'tmp/wavllm/{dataset_name}/wavllm_test.tsv', 'a') as f:
                f.write(f"{chunk_index}\t{root_path}/AudioBench/{audio_path}\t0\t{instruction}\t{answer}\tTrue\n")
            
            chunk_index += 1
            sample_chunk_number.append(1)

    # Wait for 20 seconds to ensure that the files are written and synchronized
    logger.info("Waiting for 20 seconds to ensure that the files are written and synchronized.")
    time.sleep(20)

    # Running a Bash command to run the inference
    logger.info("Running the WavLLM model inference using fairseq.")
    process = subprocess.Popen(
                    ['bash', f'{root_path}/AudioBench/src/model_src/wavllm_fairseq.sh', f'{dataset_name}'],
                    stdout=sys.stdout,
                    stderr=sys.stderr
                    )
    process.communicate()

    # Load the answers
    chunk_answers = []
    with open(f'tmp/wavllm/{dataset_name}/generate-wavllm_test.txt', 'r') as f:
        # drop the first line as it is the dummy cold start
        lines = f.readlines()[1:]
        for line in lines:
            chunk_answers.append(line.split('\t')[1].strip().strip("'").strip('"'))

    if len(chunk_answers) != sum(sample_chunk_number):
        logger.error("Number of answers does not match the number of audio chunks.")
        raise ValueError("Number of answers does not match the number of audio chunks.")

    # Aggregate the answers
    answers = []
    start   = 0
    for i in range(len(sample_chunk_number)):
        answer = ' '.join(chunk_answers[start:start+sample_chunk_number[i]])
        answers.append(answer)
        start += sample_chunk_number[i]

    return answers