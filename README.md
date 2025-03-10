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


## 📝 Change log

* *JAN 2025*: AudioBench paper is accepted to NAACL 2025 Main Conference.
* *JAN 2025*: Support 10+ [MNSC - Singlish Understanding](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1) datasets, the results are updated on leaderboard.
* *DEC 2024*: Support more (35) datasets / more Models (2 cascade and 3 fusion models).
* *SEP 2024*: Add [MuChoMusic](https://arxiv.org/abs/2408.01337) dataset for music evaluation (multiple choice questions).
* *AUG 2024*: Support a 6 speech translation datasets. Update the evaluation script for several MCQ evaluation.
* *AUG 2024*: Leaderboard is live. Check it out [here](https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard).
* *JUL 2024*: We are working hard on the leaderboard and speech translation dataset. Stay tuned!
* *JUL 2024*: Support all INITIAL 26 datasets listed in AudioBench manuscript.


[![Star History Chart](https://api.star-history.com/svg?repos=AudioLLMs/AudioBench&type=Date)](https://star-history.com/#AudioLLMs/AudioBench&Date)

## Supported Evaluation Data
- [librispeech_test_clean](./examples/supported_datasets.md)
- [librispeech_test_other](./examples/supported_datasets.md)
- [common_voice_15_en_test](./examples/supported_datasets.md)
- [peoples_speech_test](./examples/supported_datasets.md)
- [gigaspeech_test](./examples/supported_datasets.md)
- [tedlium3_test](./examples/supported_datasets.md)
- [tedlium3_long_form_test](./examples/supported_datasets.md)
- [earnings21_test](./examples/supported_datasets.md)
- [earnings22_test](./examples/supported_datasets.md)
- [aishell_asr_zh_test](./examples/supported_datasets.md)
- [covost2_en_id_test](./examples/supported_datasets.md)
- [covost2_en_zh_test](./examples/supported_datasets.md)
- [covost2_en_ta_test](./examples/supported_datasets.md)
- [covost2_id_en_test](./examples/supported_datasets.md)
- [covost2_zh_en_test](./examples/supported_datasets.md)
- [covost2_ta_en_test](./examples/supported_datasets.md)
- [cn_college_listen_mcq_test](./examples/supported_datasets.md)
- [slue_p2_sqa5_test](./examples/supported_datasets.md)
- [dream_tts_mcq_test](./examples/supported_datasets.md)
- [public_sg_speech_qa_test](./examples/supported_datasets.md)
- [spoken_squad_test](./examples/supported_datasets.md)
- [openhermes_audio_test](./examples/supported_datasets.md)
- [alpaca_audio_test](./examples/supported_datasets.md)
- [clotho_aqa_test](./examples/supported_datasets.md)
- [wavcaps_qa_test](./examples/supported_datasets.md)
- [audiocaps_qa_test](./examples/supported_datasets.md)
- [wavcaps_test](./examples/supported_datasets.md)
- [audiocaps_test](./examples/supported_datasets.md)
- [iemocap_emotion_test](./examples/supported_datasets.md)
- [meld_sentiment_test](./examples/supported_datasets.md)
- [meld_emotion_test](./examples/supported_datasets.md)
- [voxceleb_accent_test](./examples/supported_datasets.md)
- [voxceleb_gender_test](./examples/supported_datasets.md)
- [iemocap_gender_test](./examples/supported_datasets.md)
- [muchomusic_test](./examples/supported_datasets.md)
- [imda_part1_asr_test](./examples/supported_datasets.md)
- [imda_part2_asr_test](./examples/supported_datasets.md)
- [imda_part3_30s_asr_test](./examples/supported_datasets.md)
- [imda_part4_30s_asr_test](./examples/supported_datasets.md)
- [imda_part5_30s_asr_test](./examples/supported_datasets.md)
- [imda_part6_30s_asr_test](./examples/supported_datasets.md)
- [imda_part3_30s_sqa_human_test](./examples/supported_datasets.md)
- [imda_part4_30s_sqa_human_test](./examples/supported_datasets.md)
- [imda_part5_30s_sqa_human_test](./examples/supported_datasets.md)
- [imda_part6_30s_sqa_human_test](./examples/supported_datasets.md)
- [imda_part3_30s_ds_human_test](./examples/supported_datasets.md)
- [imda_part4_30s_ds_human_test](./examples/supported_datasets.md)
- [imda_part5_30s_ds_human_test](./examples/supported_datasets.md)
- [imda_part6_30s_ds_human_test](./examples/supported_datasets.md)
- [imda_ar_sentence](./examples/supported_datasets.md)
- [imda_ar_dialogue](./examples/supported_datasets.md)
- [imda_gr_sentence](./examples/supported_datasets.md)
- [imda_gr_dialogue](./examples/supported_datasets.md)
- [seame_dev_man](./examples/supported_datasets.md)
- [seame_dev_sge](./examples/supported_datasets.md)


How to evaluate with the supported datasets? That's as simple as it can be. Replace the DATASET and METRIC name.
```
DATASET=librispeech_test_clean
METRIC=wer
```

### How to Evaluation on Your Dataset?
Two simple steps:
1. Make a copy of one of the customized dataset loader. Example: [cn_college_listen_mcq_test](src/dataset_src/cn_college_listen_mcq_test.py). Customize it as your like on your own dataset.
2. Add a new term in [dataset.py](src/dataset.py).
3. Done!


## Supported Models
- [MERaLiON-AudioLLM](./examples/adding_new_model.md)
- [Whisper-large-v2](./examples/adding_new_model.md)

### How to evaluation your own models?
As long as the model can do inference, you can load them and inference to get the responses.
To evaluate on new models, please refer to [adding_new_model](./examples/adding_new_model.md).


## 🔧 Installation

Installation with pip:
```shell
pip install -r requirements.txt
```

## ⏩ Quick Start

For model-as-judge evaluation, we serve the judgement model as a service via `vllm` on port `5000`.

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



## 📖 Citation
If you find our work useful, please consider citing our paper!
```bibtex
@article{wang2024audiobench,
  title={AudioBench: A Universal Benchmark for Audio Large Language Models},
  author={Wang, Bin and Zou, Xunlong and Lin, Geyu and Sun, Shuo and Liu, Zhuohan and Zhang, Wenyu and Liu, Zhengyuan and Aw, AiTi and Chen, Nancy F},
  journal={NAACL},
  year={2025}
}
```

#### Researchers, companies or groups that are using AudioBench:
- [Llama3-S: When Llama Learns to Listen](https://homebrew.ltd/blog/llama3-just-got-ears)
- More to come...


