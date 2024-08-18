

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=salmonn_7b
GPU=3
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=20
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


DATASET=openhermes_audio_test
METRICS=llama3_70b_judge
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=alpaca_audio_test
METRICS=llama3_70b_judge
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES
