


DATASET=voxceleb_gender_test
METRICS=llama3_70b_judge_binary
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=iemocap_gender_test
METRICS=llama3_70b_judge_binary
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES
