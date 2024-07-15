

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct
GPU=2
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=salmonn_7b
GPU=2
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# SQA
DATASET=cn_college_listen_test
DATASET=slue_p2_sqa5_test
DATASET=public_sg_speech_qa_test
DATASET=dream_tts_test

METRICS=llama3_70b_judge



# ASR
DATASET=librispeech_test_clean
DATASET=librispeech_test_other
DATASET=common_voice_15_en_test
DATASET=peoples_speech_test

METRICS=wer



bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES