export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
export NLTK_DATA="/project/cache/nltk_data"


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_NAME=gpt-4o-audio

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
GPU=0
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=-1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


DATASET=mmau_mini
METRICS=llama3_70b_judge

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

