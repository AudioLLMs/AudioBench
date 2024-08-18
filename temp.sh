

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=salmonn_7b
GPU=3
BATCH_SIZE=1
OVERWRITE=False
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# ASR
DATASET=librispeech_test_clean
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=librispeech_test_other
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=common_voice_15_en_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=peoples_speech_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=gigaspeech_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=earnings21_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=earnings22_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=tedlium3_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

DATASET=tedlium3_long_form_test
METRICS=wer
bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

<<COMMENT

# SQA
DATASET=cn_college_listen_test
DATASET=slue_p2_sqa5_test
DATASET=public_sg_speech_qa_test
DATASET=dream_tts_test

METRICS=llama3_70b_judge


# SI
DATASET=openhermes_audio_test
DATASET=alpaca_audio_test

METRICS=llama3_70b_judge


# AC
DATASET=audiocaps_test
DATASET=wavcaps_test

METRICS=llama3_70b_judge
METRICS=meteor


# ASQA
DATASET=clotho_aqa_test
DATASET=audiocaps_qa_test
DATASET=wavcaps_qa_test

METRICS=llama3_70b_judge


# AR
DATASET=voxceleb_accent_test

METRICS=llama3_70b_judge


# GR
DATASET=voxceleb_gender_test
DATASET=iemocap_gender_test


METRICS=llama3_70b_judge


# ER
DATASET=iemocap_emotion_test
DATASET=meld_sentiment_test
DATASET=meld_emotion_test

METRICS=llama3_70b_judge

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES


COMMENT