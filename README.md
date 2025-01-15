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

* *JAN 2025*: Support 10+ [MNSC - Singlish Understanding](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1) datasets.
* *DEC 2024*: Support More  (35) datasets / More Models (2 cascade and 3 fusion models).
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

MODEL_NAME=Qwen2-Audio-7B-Instruct

DATASET=cn_college_listen_mcq_test
METRICS=llama3_70b_judge_binary

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

```

## How to Evaluation AudioBench Supported Datasets?

That's as simple as it can be. Replace the DATASET and METRIC name. A full list of supported datasets can be found: [SUPPORTED DATASETS](./examples/supported_datasets.md).
```
DATASET=librispeech_test_clean
METRIC=wer
```

## How to Evaluation AudioBench Supported Models?


That's as simple as it can be. Replace the MODEL_NAME. A full list of supported datasets can be found: [SUPPORTED MODELS](./examples/supported_models.md).
```
MODEL_NAME=cascade_whisper_large_v3_llama_3_8b_instruct
```


## How to Evaluation Your Models?
To evaluate on new models, please refer to [adding_new_model](./examples/adding_new_model.md).

## How to Evaluation on Your Dataset?
Two simple steps:
1. Add dataset loader and inference part. Example for [cn_college_listen_mcq_test](src/dataset_src/cn_college_listen_mcq_test.py)
2. Edit [dataset.py](src/dataset.py)



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


