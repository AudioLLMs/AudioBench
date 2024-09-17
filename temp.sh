

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=temp_model
GPU=0
BATCH_SIZE=1
OVERWRITE=False
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



DATASET=mu_chomusic_test
METRICS=llama3_70b_judge_binary
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES
