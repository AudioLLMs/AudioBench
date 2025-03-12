
Supported datasets:


```

# == == == == == ASR English == == == == ==

DATASET=librispeech_test_clean
METRIC=wer

DATASET=librispeech_test_other
METRIC=wer

DATASET=common_voice_15_en_test
METRIC=wer

DATASET=peoples_speech_test
METRIC=wer

DATASET=gigaspeech_test
METRIC=wer

DATASET=tedlium3_test
METRIC=wer

DATASET=tedlium3_long_form_test
METRIC=wer

DATASET=earnings21_test
METRIC=wer

DATASET=earnings22_test
METRIC=wer




# == == == == == ASR - Singlish == == == == ==

DATASET=imda_part1_asr_test
METRIC=wer

DATASET=imda_part2_asr_test
METRIC=wer

DATASET=imda_part3_30s_asr_test
METRIC=wer

DATASET=imda_part4_30s_asr_test
METRIC=wer

DATASET=imda_part5_30s_asr_test
METRIC=wer

DATASET=imda_part6_30s_asr_test
METRIC=wer


# == == == == == ASR - Mandarin == == == == ==

DATASET=aishell_asr_zh_test
METRIC=wer



# == == == == == Speech Translation == == == == ==

DATASET=covost2_en_id_test
METRIC=bleu

DATASET=covost2_en_zh_test
METRIC=bleu

DATASET=covost2_en_ta_test
METRIC=bleu

DATASET=covost2_id_en_test
METRIC=bleu

DATASET=covost2_zh_en_test
METRIC=bleu

DATASET=covost2_ta_en_test
METRIC=bleu



# == == == == == Spoken Question Answering == == == == ==

DATASET=cn_college_listen_mcq_test
METRIC=llama3_70b_judge

DATASET=slue_p2_sqa5_test
METRIC=llama3_70b_judge

DATASET=dream_tts_mcq_test
METRIC=llama3_70b_judge

DATASET=public_sg_speech_qa_test
METRIC=llama3_70b_judge

DATASET=spoken_squad_test
METRIC=llama3_70b_judge

# Singlish SQA

DATASET=imda_part3_30s_sqa_human_test
METRIC=llama3_70b_judge

DATASET=imda_part4_30s_sqa_human_test
METRIC=llama3_70b_judge

DATASET=imda_part5_30s_sqa_human_test
METRIC=llama3_70b_judge

DATASET=imda_part6_30s_sqa_human_test
METRIC=llama3_70b_judge


# == == == == == Spoken Dialogue Summarization == == == == ==


DATASET=imda_part3_30s_ds_human_test
METRIC=llama3_70b_judge

DATASET=imda_part4_30s_ds_human_test
METRIC=llama3_70b_judge

DATASET=imda_part5_30s_ds_human_test
METRIC=llama3_70b_judge

DATASET=imda_part6_30s_ds_human_test
METRIC=llama3_70b_judge


# == == == == == Speech Instruction == == == == ==

DATASET=openhermes_audio_test
METRIC=llama3_70b_judge

DATASET=alpaca_audio_test
METRIC=llama3_70b_judge




# == == == == == Audio Scene Question Answering == == == == ==

DATASET=clotho_aqa_test
METRIC=llama3_70b_judge

DATASET=wavcaps_qa_test
METRIC=llama3_70b_judge

DATASET=audiocaps_qa_test
METRIC=llama3_70b_judge




# == == == == == Audio Captioning == == == == ==

DATASET=wavcaps_test
METRIC=llama3_70b_judge

DATASET=wavcaps_test
METRIC=meteor

DATASET=audiocaps_test
METRIC=llama3_70b_judge

DATASET=audiocaps_test
METRIC=meteor



# == == == == == Emotion Recognition == == == == ==

DATASET=iemocap_emotion_test
METRIC=llama3_70b_judge

DATASET=meld_sentiment_test
METRIC=llama3_70b_judge

DATASET=meld_emotion_test
METRIC=llama3_70b_judge




# == == == == == Accent Recognition == == == == ==

DATASET=voxceleb_accent_test
METRIC=llama3_70b_judge

DATASET=imda_ar_sentence
METRIC=llama3_70b_judge

DATASET=imda_ar_dialogue
METRIC=llama3_70b_judge

# == == == == == Gender Recognition == == == == ==

DATASET=voxceleb_gender_test
METRIC=llama3_70b_judge

DATASET=iemocap_gender_test
METRIC=llama3_70b_judge

DATASET=imda_gr_sentence
METRIC=llama3_70b_judge

DATASET=imda_gr_dialogue
METRIC=llama3_70b_judge

# == == == == == Music Understanding == == == == ==

DATASET=mu_chomusic_test
METRIC=llama3_70b_judge

# == == == == == ASR Code-Switching == == == == ==

# SEAME dataset for Madarine-English code-switching with Singapore accent.
#Lyu, Dau-Cheng, Tien Ping Tan, Engsiong Chng, and Haizhou Li. "SEAME: a Mandarin-English code-switching speech corpus in south-east asia." In Interspeech, vol. 10, pp. 1986-1989. 2010.

DATASET=seame_dev_man
METRIC=wer

DATASET=seame_dev_sge
METRIC=wer

```


