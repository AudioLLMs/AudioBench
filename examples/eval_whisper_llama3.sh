

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
export MODEL_NAME=whisper_large_v3_with_llama_3_8b_instruct
export GPU=3
export BATCH_SIZE=1
export OVERWRITE=False
export NUMBER_OF_SAMPLES=-1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# ASR
bash examples/eval_asr_en.sh

# SQA
bash examples/eval_sqa.sh

# SI
bash examples/eval_si.sh

# ST
bash examples/eval_st.sh

# ASQA
bash examples/eval_asqa.sh

# AC
bash examples/eval_ac.sh

# ER
bash examples/eval_er.sh

# AR
bash examples/eval_ar.sh

# GR
bash examples/eval_gr.sh

# ASR-CN
bash examples/eval_asr_cn.sh

