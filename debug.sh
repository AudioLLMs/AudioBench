export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
export NLTK_DATA="/project/cache/nltk_data"


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_NAME=gemini-1.5-flash

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
GPU=1
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=-1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


DATASET=mmau_mini
METRICS=string_match

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

