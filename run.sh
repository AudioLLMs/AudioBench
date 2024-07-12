

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct
GPU=2
BATCH_SIZE=1
METRICS=llama3_70b_judge
OVERWRITE=True
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=salmonn_7b
GPU=2
BATCH_SIZE=1
METRICS=llama3_70b_judge
OVERWRITE=True
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

DATASET=cn_college_listen_test
DATASET=slue_p2_sqa5_test

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES