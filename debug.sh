export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
export NLTK_DATA="/project/cache/nltk_data"

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Tested
# MODEL_NAME=cascade_whisper_large_v3_llama_3_8b_instruct
# MODEL_NAME=cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct
# MODEL_NAME=Qwen2-Audio-7B-Instruct
# MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION
# MODEL_NAME=whisper_large_v3
# MODEL_NAME=Qwen-Audio-Chat
# MODEL_NAME=SALMONN_7B
# MODEL_NAME=WavLLM_fairseq
# MODEL_NAME=gemini-1.5-flash


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_NAME=whisper_large_v3

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
GPU=1
BATCH_SIZE=1
OVERWRITE=False
NUMBER_OF_SAMPLES=20
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

DATASET=imda_part4_30s_asr_test
METRICS=wer

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

