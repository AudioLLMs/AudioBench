


DATASET=cn_college_listen_mcq_test
METRICS=llama3_70b_judge_binary
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=slue_p2_sqa5_test
METRICS=llama3_70b_judge
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=dream_tts_mcq_test
METRICS=llama3_70b_judge_binary
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=public_sg_speech_qa_test
METRICS=llama3_70b_judge
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=spoken_squad_test
METRICS=llama3_70b_judge
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES
