<p align="center">
  <img src="assets/logo.png" alt="Prometheus-Logo" style="width: 15%; display: block; margin: auto;">
</p>

<h1 align="center">🔥 AudioBench 🔥</h1>



<p align="center">
  <a href="https://arxiv.org/abs/2406.16020"><img src="https://img.shields.io/badge/arXiv-2406.16020-b31b1b.svg" alt="arXiv"></a>
  <a href="AudioLLMs"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard"><img src="https://img.shields.io/badge/AudioBench-Leaderboard-g41b1b.svg" alt="License"></a>
</p>

<p align="center">
  ⚡ A repository for evaluating AudioLLMs in various tasks 🚀 ⚡ <br>
  ⚡ AudioBench: A Universal Benchmark for Audio Large Language Models 🚀 ⚡ <br>
</p>



## 🔧 Installation

Installation with pip:
```shell
pip install -r requirements.txt
```
For model-as-judge evaluation, we serve the judgement model as a service via `vllm` on port `5000`.


## ⏩ Quick Start

The example is hosting a `Llama-3-70B-Instruct` model and running the cascade `Whisper + Llama-3` model.
```shell
# Step 1:
# Server the model as judge
# It will auto-download the model and may requires verification from Hugging Face.
# In the demo, we use 2 H100 80G in order to host the model.
# For smaller VRAM, you may need to reduce the model size.
bash host_model_judge_llama_3_70b_instruct.sh

# Step 2:
# The example is done with 3 H100 80G GPUs.
# The AudioLLMs model inference is done on GPU 2 since GPU 0&1 is used to host model-as-judge services.
# This is a setting for just using 50 samples for evaluation.
MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct
GPU=2
BATCH_SIZE=1
METRICS=llama3_70b_judge
OVERWRITE=True
NUMBER_OF_SAMPLES=50

DATASET=cn_college_listen_test

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

# Step 3:
# The results would be like:
#    "llama3_70b_judge": {
#        "judge_score": 3.12,
#        "success_rate": 1.0
#    }

```
The example is how to get started. To evaluate on the full datasets, please refer to [Examples](./examples/).


## 📚 Supported Models and Datasets

### Datasets
```
SU=Speech Understanding
  ASR=Automatic Speech Recognition
  SQA=Speech Question Answering
  SI=Speech Instruction

ASU=Audio Scene Understanding
  AC=Audio Captioning
  ASQA=Audio Scene Question Answering

VU=Voice Understanding
  AR=Accent Recognition
  GR=Gender Recognition
  ER=Emotion Recognition
```

|Dataset|Category|Task|Metrics|Status|
|---|---|---|---|---|
|**LibriSpeech-Clean**|SU|ASR|WER|✅︎|
|**LibriSpeech-Other**|SU|ASR|WER|✅︎|
|**CommonVoice-15-EN**|SU|ASR|WER|✅︎|
|**Peoples-Speech**|SU|ASR|WER|✅︎|
|**GigaSpeech**|SU|ASR|WER|✅︎|
|**Earning21**|SU|ASR|WER|✅︎|
|**Earning22**|SU|ASR|WER|✅︎|
|**Tedlium3**|SU|ASR|WER|✅︎|
|**Tedlium3-Longform**|SU|ASR|WER|✅︎|
|**CN-College-Listen**|SU|SQA|Model-as-Judge|✅︎|
|**SLUE-P2-SQA5**|SU|SQA|Model-as-Judge|✅︎|
|**Public-SG-SpeechQA**|SU|SQA|Model-as-Judge|✅︎|
|**DREAM-TTS**|SU|SQA|Model-as-Judge|✅︎|
|**OpenHermes-Audio**|SU|SI|Model-as-Judge|✅︎|
|**ALPACA-Audio**|SU|SI|Model-as-Judge|✅︎|
|**AudioCaps**|ASU|AC|Model-as-Judge / METEOR|✅︎|
|**WavCaps**|ASU|AC|Model-as-Judge / METEOR|✅︎|
|**Clotho-AQA**|ASU|ASQA|Model-as-Judge|✅︎|
|**AudioCaps-QA**|ASU|ASQA|Model-as-Judge|✅︎|
|**WavCaps-QA**|ASU|ASQA|Model-as-Judge|✅︎|
|**VoxCeleb-Accent**|VU|AR|Model-as-Judge|✅︎|
|**VoxCeleb-Gender**|VU|GR|Model-as-Judge|✅︎|
|**IEMOCAP-Gender**|VU|GR|Model-as-Judge|✅︎|
|**IEMOCAP-Emotion**|VU|ER|Model-as-Judge|✅︎|
|**MELD-Sentiment**|VU|ER|Model-as-Judge|✅︎|
|**MELD-Emotion**|VU|ER|Model-as-Judge|✅︎|


### Models
|Model|Size|Notes|Status|
|---|---|---|---|
|Whisper-Large + Llama-3-8B-Instruct|~8B|Cascade Models|✅︎|
|SALMONN-7B|~7B|AudioLLM - Fusion Model|✅︎|
|Qwen-Audio|~8B|AudioLLM - Fusion Model|TODO|


## 📖 Citation
If you find our work useful, please consider citing our paper!
```bibtex
@article{wang2024audiobench,
  title={AudioBench: A Universal Benchmark for Audio Large Language Models},
  author={Wang, Bin and Zou, Xunlong and Lin, Geyu and Sun, Shuo and Liu, Zhuohan and Zhang, Wenyu and Liu, Zhengyuan and Aw, AiTi and Chen, Nancy F},
  journal={arXiv preprint arXiv:2406.16020},
  year={2024}
}
```

