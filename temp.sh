

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=temp_model
GPU=3
BATCH_SIZE=1
OVERWRITE=False
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



DATASET=aishell_asr_zh_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES
