
# export HF_ENDPOINT=https://huggingface.co

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
export NLTK_DATA="/project/cache/nltk_data"


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# MODEL_NAME=multitask-subsetv2:whisper_specaugment+seqcnn8+lora:f3epoch:run2.0:4gpu
# MODEL_NAME=audiollm_imda
# MODEL_NAME=huayun_whisper_local_cs
# MODEL_NAME=huayun_whisper_local_no_cs
# MODEL_NAME=xl_whisper_imda_v0_1
# MODEL_NAME=original_whisper_large_v2
# MODEL_NAME=MERaLiON_AudioLLM_v0_5
# MODEL_NAME=MERaLiON_AudioLLM_v0_5_v2
# MODEL_NAME=MERaLiON_AudioLLM_v0_5_average5
# MODEL_NAME=MERaLiON_AudioLLM_v0_5_average5_better_asr
# MODEL_NAME=MERaLiON_AudioLLM_v1
# MODEL_NAME=cascade_whisper_large_v3_llama_3_8b_instruct
# MODEL_NAME=Qwen2-Audio-7B-Instruct
# MODEL_NAME=Qwen-Audio-Chat
# MODEL_NAME=SALMONN_7B
# MODEL_NAME=AudioLLM_IMDA_MLP100
# MODEL_NAME=temp_model_for_debugging_datasets
# MODEL_NAME=WavLLM_fairseq

# Tested
# MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION
# MODEL_NAME=cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct
# MODEL_NAME=gemini-1.5-flash

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
GPU=1
BATCH_SIZE=1
OVERWRITE=True
# NUMBER_OF_SAMPLES=-1
NUMBER_OF_SAMPLES=10
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# DATASET=mediacorp_short_test
# METRICS=wer

# DATASET=cn_college_listen_mcq_test
# METRICS=llama3_70b_judge_binary

# DATASET=imda_part6_30s_asr_test
# METRICS=wer

# DATASET=muchomusic_test
# METRICS=llama3_70b_judge_binary

# DATASET=ytb_asr_batch1
# METRICS=wer

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


DATASET=ytb_asr_batch2
METRICS=wer

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES


