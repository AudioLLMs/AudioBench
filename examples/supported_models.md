
Supported models:


```
# This is a cascaded method with Whisper-large-v3 and LLAMA-3-8B-Instruct
MODEL_NAME=cascade_whisper_large_v3_llama_3_8b_instruct

# This is a cascaded method with Whisper-large-v2 and SEALION-V3 LLM model.
MODEL_NAME=cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct

# The Qwen2-Audio Model: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
MODEL_NAME=Qwen2-Audio-7B-Instruct

# The Qwen-Audio Model: https://huggingface.co/Qwen/Qwen-Audio-Chat
MODEL_NAME=Qwen-Audio-Chat

# This is the SALMONN model: https://arxiv.org/abs/2310.13289
MODEL_NAME=SALMONN_7B

# MERaLiON-AudioLLM: https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION
MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION

# Only whisper - for ASR / ST Tasks
MODEL_NAME=whisper_large_v3
MODEL_NAME=whisper_large_v2

```


## Preparation for SALMONN_7B

```
# Move to examples folder
cd examples
# need Git LFS to download large model files
# e.g. apt install git-lfs
git clone https://huggingface.co/AudioLLMs/SALMONN_7B

cd ..
bash examples/eval_SALMONN_7B.sh
```



