export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
export NLTK_DATA="/project/cache/nltk_data"


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_NAME=seallms_audio_7b

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
GPU=1
BATCH_SIZE=1
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



DATASET=gigaspeech2_viet
METRICS=wer

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

