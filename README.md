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

## 📚 Supported Models and Datasets

### Datasets
|Dataset|Category|Task|Size|Metrics|ID|Status|
|---|---|---|---|---|---|---|
|**CN-College-Listen**|Speech Understanding|Speech Question Answering|~2200|Model-as-Judge|cn_college_listen_test|✅︎|
|**SLUE-P2-SQA5**|||~400|Model-as-Judge|slue_p2_sqa5_test|✅︎|
|**Public-SG-SpeechQA**|||688|Model-as-Judge|public_sg_speech_qa_test|✅︎|


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


