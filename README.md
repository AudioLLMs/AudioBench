<p align="center">
  <img src="assets/logo.png" alt="Prometheus-Logo" style="width: 15%; display: block; margin: auto;">
</p>

<h1 align="center">🔥 AudioBench 🔥</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2406.16020"><img src="https://img.shields.io/badge/arXiv-2406.16020-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/AudioLLMs"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard"><img src="https://img.shields.io/badge/AudioBench-Leaderboard-g41b1b.svg" alt="License"></a>
</p>

<p align="center">
  ⚡ A repository for evaluating AudioLLMs in various tasks 🚀 ⚡ <br>
  ⚡ AudioBench: A Universal Benchmark for Audio Large Language Models 🚀 ⚡ <br>
</p>

## Change log

* *Aug 2024*: Support a couple of speech translation datasets. Update the evaluation script for several MCQ evaluation.
* *Aug 2024*: Leadboard is live. Check it out [here](https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard).
* *July 2024*: We are working hard on the leaderboard and speech translation dataset. Stay tuned!
* *July 2024*: Support all 26 datasets listed in AudioBench manuscript.



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
# bash host_model_judge_llama_3_70b_instruct.sh

# Another option (recommended) is to use the quantized model which could be hosted on 2*40G GPUs.
bash host_model_judge_llama_3_70b_instruct_awq.sh

# Step 2:
# The example is done with 3 H100 80G GPUs.
# The AudioLLMs model inference is done on GPU 2 since GPU 0&1 is used to host model-as-judge services.
# This is a setting for just using 50 samples for evaluation.
MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct
GPU=2
BATCH_SIZE=1
METRICS=llama3_70b_judge_binary
OVERWRITE=True
NUMBER_OF_SAMPLES=50

DATASET=cn_college_listen_mcq_test

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

# Step 3:
# The results would be like:
#    "llama3_70b_judge_binary": {
#        "judge_score": 90.0,
#        "success_rate": 1.0
#    }
#}
# This indicates that the cascade model can achieve 90% accuracy on the MCQ task for English listening test.

```
The example is how to get started. To evaluate on the full datasets, please refer to [Examples](./examples/).

```shell
# After model weight download, run the evaluation script for all datasets
bash examples/eval_salmonn_7b.sh
```


## 📚 Supported Models and Datasets

### Datasets

### Speech Understanding
- **ASR**: [Automatic Speech Recognition](#ASR-English)
- **SQA**: [Speech Question Answering](#SQA)
- **SI**: [Speech Instruction](#SI)
- **ST**: [Speech Translation](#ST)
- **ASR-CN**: [Automatic Speech Recognition for Chinese](#ASR-Chinese)

### Audio Scene Understanding
- **AC**: [Audio Captioning](#AC)
- **ASQA**: [Audio Scene Question Answering](#ASQA)

### Voice Understanding
- **AR**: [Accent Recognition](#AR)
- **GR**: [Gender Recognition](#GR)
- **ER**: [Emotion Recognition](#ER)


#### ASR-English
|Dataset|Metrics|Status|
|---|---|---|
|**LibriSpeech-Clean**|Word-Error-Rate|✅|
|**LibriSpeech-Other**|Word-Error-Rate|✅|
|**CommonVoice-15-EN**|Word-Error-Rate|✅|
|**Peoples-Speech**|Word-Error-Rate|✅|
|**GigaSpeech**|Word-Error-Rate|✅|
|**Earning21**|Word-Error-Rate|✅|
|**Earning22**|Word-Error-Rate|✅|
|**Tedlium3**|Word-Error-Rate|✅|
|**Tedlium3-Longform**|Word-Error-Rate|✅|

```shell
export MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct
export GPU=3
export BATCH_SIZE=1
export OVERWRITE=False
export NUMBER_OF_SAMPLES=-1
bash examples/eval_sqa.sh
```

#### SQA
|Dataset|Metrics|Status|
|---|---|---|
|**CN-College-Listen**|Model-as-Judge (binary)|✅|
|**SLUE-P2-SQA5**|Model-as-Judge|✅|
|**DREAM-TTS**|Model-as-Judge (binary)|✅|
|**Public-SG-SpeechQA**|Model-as-Judge|✅|
|**Spoken-SQuAD**|Model-as-Judge|✅|

```shell
bash examples/eval_sqa.sh
```

#### SI
|Dataset|Metrics|Status|
|---|---|---|
|**OpenHermes-Audio**|Model-as-Judge|✅|
|**ALPACA-Audio**|Model-as-Judge|✅|

```shell
bash examples/eval_si.sh
```

#### ST
|Dataset|Metrics|Status|
|---|---|---|
|**CoVost2-English-Indonesian**|BLEU|✅|
|**CoVost2-English-Chinese**|BLEU|✅|
|**CoVost2-English-Tamil**|BLEU|✅|
|**CoVost2-Indonesian-English**|BLEU|✅|
|**CoVost2-Chinese-English**|BLEU|✅|
|**CoVost2-Tamil-English**|BLEU|✅|

```shell
bash examples/eval_st.sh
```

#### ASR-Chinese
|Dataset|Metrics|Status|
|---|---|---|
|**AISHELL-ASR-ZH**|Word-Error-Rate|✅|

```shell
bash examples/eval_asr_cn.sh
```

#### AC
|Dataset|Metrics|Status|
|---|---|---|
|**AudioCaps**|Model-as-Judge / METEOR|✅|
|**WavCaps**|Model-as-Judge / METEOR|✅|

```shell
bash examples/eval_ac.sh
```

#### ASQA
|Dataset|Metrics|Status|
|---|---|---|
|**Clotho-AQA**|Model-as-Judge|✅|
|**AudioCaps-QA**|Model-as-Judge|✅|
|**WavCaps-QA**|Model-as-Judge|✅|

```shell
bash examples/eval_asqa.sh
```

#### AR
|Dataset|Metrics|Status|
|---|---|---|
|**VoxCeleb-Accent**|Model-as-Judge|✅|

```shell
bash examples/eval_ar.sh
```

#### GR
|Dataset|Metrics|Status|
|---|---|---|
|**VoxCeleb-Gender**|Model-as-Judge (binary)|✅|
|**IEMOCAP-Gender**|Model-as-Judge (binary)|✅|

```shell
bash examples/eval_gr.sh
```

#### ER
|Dataset|Metrics|Status|
|---|---|---|
|**IEMOCAP-Emotion**|Model-as-Judge (binary)|✅|
|**MELD-Sentiment**|Model-as-Judge (binary)|✅|
|**MELD-Emotion**|Model-as-Judge (binary)|✅|

```shell
bash examples/eval_er.sh
```

### Models
|Name|Size|Notes|Status|
|---|---|---|---|
|Whisper-Large+Llama-3-8B-Instruct|~8B|Cascade Models|✅|
|SALMONN|~7B|End2End|✅|
|Qwen-Audio|~8B|End2End|TODO|
|WavLM|~7B|End2End|TODO|
|Qwen2-Audio|~8B|End2End|TODO|

More models are accessible in this [survey]((https://github.com/AudioLLMs/AudioLLM)).
To add a new model, please refer to [Adding a New Model](./examples/adding_new_model.md).



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





