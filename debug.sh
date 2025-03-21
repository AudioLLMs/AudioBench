export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
export NLTK_DATA="/project/cache/nltk_data"


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
GPU=0
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=10
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


DATASET=cn_college_listen_mcq_test
METRICS=llama3_70b_judge

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

