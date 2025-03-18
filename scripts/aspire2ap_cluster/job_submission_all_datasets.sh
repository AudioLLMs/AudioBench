

MODEL_NAME=MERaLiON-AudioLLM-Whisper-SEA-LION


echo "MODEL_NAME: $MODEL_NAME"

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# # ASR
# qsub -v DATASET_NAME=librispeech_test_clean,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=librispeech_test_other,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=common_voice_15_en_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=peoples_speech_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=gigaspeech_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=tedlium3_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=tedlium3_long_form_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=earnings21_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=earnings22_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=aishell_asr_zh_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh

# # ST
# qsub -v DATASET_NAME=covost2_en_id_test,METRICS=bleu,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=covost2_en_zh_test,METRICS=bleu,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=covost2_en_ta_test,METRICS=bleu,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=covost2_id_en_test,METRICS=bleu,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=covost2_zh_en_test,METRICS=bleu,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=covost2_ta_en_test,METRICS=bleu,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh

# # SQA
# qsub -v DATASET_NAME=cn_college_listen_mcq_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=slue_p2_sqa5_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=dream_tts_mcq_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=public_sg_speech_qa_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=spoken_squad_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh

# # SI
# qsub -v DATASET_NAME=openhermes_audio_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=alpaca_audio_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # ASQA
# qsub -v DATASET_NAME=clotho_aqa_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=wavcaps_qa_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=audiocaps_qa_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # # # # AC
# qsub -v DATASET_NAME=wavcaps_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=wavcaps_test,METRICS=meteor,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=audiocaps_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=audiocaps_test,METRICS=meteor,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # Emotion
# qsub -v DATASET_NAME=iemocap_emotion_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=meld_sentiment_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=meld_emotion_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # # Accent
# qsub -v DATASET_NAME=voxceleb_accent_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # # Gender
# qsub -v DATASET_NAME=voxceleb_gender_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=iemocap_gender_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # # MUSIC
# qsub -v DATASET_NAME=muchomusic_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # IMDA-ASR
# qsub -v DATASET_NAME=imda_part1_asr_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part2_asr_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part3_30s_asr_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part4_30s_asr_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part5_30s_asr_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part6_30s_asr_test,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # IMDA-SQA
# qsub -v DATASET_NAME=imda_part3_30s_sqa_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part4_30s_sqa_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part5_30s_sqa_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part6_30s_sqa_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # IMDA-SDS
# qsub -v DATASET_NAME=imda_part3_30s_ds_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part4_30s_ds_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part5_30s_ds_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_part6_30s_ds_human_test,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# # IMDA-Paralingual
# qsub -v DATASET_NAME=imda_ar_sentence,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_ar_dialogue,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_gr_sentence,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=imda_gr_dialogue,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh

# qsub -v DATASET_NAME=mmau_mini,METRICS=llama3_70b_judge,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=mmau_mini,METRICS=string_match,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# qsub -v DATASET_NAME=seame_dev_man,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=seame_dev_sge,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh


# qsub -v DATASET_NAME=gigaspeech2_thai,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=gigaspeech2_indo,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=gigaspeech2_viet,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh




# qsub -v DATASET_NAME=ytb_asr_batch1,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh
# qsub -v DATASET_NAME=ytb_asr_batch2,METRICS=wer,MODEL_NAME=$MODEL_NAME scripts/aspire2ap_cluster/job_submission.sh










