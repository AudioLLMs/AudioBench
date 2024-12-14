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
    <a href="https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard" 
     target="_blank" 
     style="text-decoration: none; color: #0078d7; font-weight: bold; font-size: 18px;">
     🌟 Come to View Our Live Leaderboard on Huggingface Space 🌟
  </a>
</p>


## Change log

* *DEC 2024*: Add [MuChoMusic](https://arxiv.org/abs/2408.01337) dataset for music evaluation (multiple choice questions).
* *SEP 2024*: Add [MuChoMusic](https://arxiv.org/abs/2408.01337) dataset for music evaluation (multiple choice questions).
* *AUG 2024*: Support a 6 speech translation datasets. Update the evaluation script for several MCQ evaluation.
* *AUG 2024*: Leaderboard is live. Check it out [here](https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard).
* *JUL 2024*: We are working hard on the leaderboard and speech translation dataset. Stay tuned!
* *JUL 2024*: Support all INITIAL 26 datasets listed in AudioBench manuscript.



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
# Server the judgement model using VLLM framework (my example is using int4 quantized version)
# This requires with 1 * 80GB GPU
bash vllm_model_judge_llama_3_70b.sh

# Step 2:
# We perform model inference and obtain the evaluation results with the second GPU
GPU=2
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=-1 # indicate all test samples if number_of_samples=-1

MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct

DATASET=cn_college_listen_mcq_test
METRICS=llama3_70b_judge_binary

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES


```shell
# After model weight download, run the evaluation script for all datasets
bash examples/eval_salmonn_7b.sh
```

# How to Evaluation Supported Datasets in AudioBench?

That's as simple as it can be. Replace the DATASET and METRIC name. A full list of supported datasets can be found: [SUPPORTED DATASETS](./examples/supported_datasets.md).

```
DATASET=librispeech_test_clean
METRIC=wer
```

# How to Evaluation Your Models?
The example is how to get started. To evaluate on the other datasets, please refer to [Examples](./examples/).


# How to Add New Evaluation Data and Associated Metric?


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

#### Music
|Dataset|Metrics|Status|
|---|---|---|
|**MuChoMusic**|Model-as-Judge (binary)|✅|


```shell
bash examples/eval_music.sh
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

#### Researchers, companies or groups that are using AudioBench:
- [Llama3-S: When Llama Learns to Listen](https://homebrew.ltd/blog/llama3-just-got-ears)
- More to come...


