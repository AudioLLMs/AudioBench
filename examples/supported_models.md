
Supported models:


```
MODEL_NAME=cascade_whisper_large_v3_llama_3_8b_instruct

MODEL_NAME=cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct

MODEL_NAME=Qwen2-Audio-7B-Instruct

MODEL_NAME=SALMONN_7B

MODEL_NAME=Qwen-Audio-Chat

MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION

MODEL_NAME=whisper_large_v3

```


## Preparation for SALMONN_7B

```
cd examples
# need Git LFS to download large model files
# e.g. apt install git-lfs
git clone https://huggingface.co/AudioLLMs/SALMONN_7B

cd ..
bash examples/eval_SALMONN_7B.sh
```



