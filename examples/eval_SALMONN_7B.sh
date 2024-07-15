



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=salmonn_7b
GPU=2
BATCH_SIZE=1
METRICS=llama3_70b_judge
OVERWRITE=True
NUMBER_OF_SAMPLES=-1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# SQA
DATASET=cn_college_listen_test
DATASET=slue_p2_sqa5_test
DATASET=public_sg_speech_qa_test
DATASET=dream_tts_test


bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES