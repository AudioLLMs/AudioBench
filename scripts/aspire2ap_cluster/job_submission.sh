#!/bin/bash
#PBS -N wb_ab_job
#PBS -l select=1:ncpus=16:ngpus=1:mem=256gb:container_engine=enroot
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -k oed
#PBS -q normal
#PBS -P 13003558
#PBS -l container_image=/data/projects/13003558/wangb1/workspaces/containers/customized_containers/audiobench_for_meralion_audiollm.sqsh
#PBS -l container_name=audiobench
#PBS -l enroot_env_file=/data/projects/13003558/wangb1/workspaces/MERaLiON-AudioLLM/scripts/nscc2/env.conf

# audiobench_v7_for_qwen_audio
# audiobench_for_meralion_audiollm
# audiobench_for_qwen_audio_chat.sqsh
# audiobench_for_wavllm
# audiobench_for_gemini
# audiobench_for_phi4

# HF
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/project/cache/huggingface_cache
NLTK_DATA=/project/cache/nltk_data

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
#If the log exist, then exit, no need to proceed with the job
if [ -f "/data/projects/13003558/wangb1/workspaces/AudioBench/log_for_all_models/${MODEL_NAME}/${DATASET_NAME}_${METRICS}_score.json" ]; then
    # echo the log filename
    echo ""
    echo "/data/projects/13003558/wangb1/workspaces/AudioBench/log_for_all_models/${MODEL_NAME}/${DATASET_NAME}_${METRICS}_score.json"
    echo "The log file exists, no need to proceed with the job."
    exit 0
fi




# Skip VLLM Server setup if METRICS is gpt4o_judge
if [ "$METRICS" == "llama3_70b_judge" ]; then
    echo "METRICS is llama3_70b_judge, initialize VLLM Server."

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # Start the VLLM Server as the judge
    MIN=5000
    MAX=6000

    MY_VLLM_PORT_JUDGE=$(( RANDOM % (MAX - MIN + 1) + MIN ))
    export MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE
    echo "VLLM PORT: $MY_VLLM_PORT_JUDGE"

    enroot create -n vllm /data/projects/13003558/wangb1/workspaces/containers/customized_containers/vllm.sqsh

    enroot start \
        -r -w \
        -m /data/projects/13003558/wangb1/workspaces:/project \
        -e NLTK_DATA=$NLTK_DATA \
        -e HF_HOME=$HF_HOME \
        -e HF_ENDPOINT=$HF_ENDPOINT \
        -e MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE \
        vllm \
        bash -c "
        cd ~
        pwd
        cd /project/AudioBench
        pwd
        bash scripts/aspire2ap_cluster/nscc2_vllm.sh > ${PBS_JOBID}.log 2>&1
        "
fi



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Start the AudioBench Evaluation
enroot start \
	-r -w \
	-m /data/projects/13003558/wangb1/workspaces:/project \
	-e NLTK_DATA=$NLTK_DATA \
	-e HF_HOME=$HF_HOME \
    -e HF_ENDPOINT=$HF_ENDPOINT \
    -e MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE \
	audiobench \
	bash -c "
    cd ~
    pwd
    cd /project/AudioBench
    pwd
    bash scripts/aspire2ap_cluster/nscc2_audiobench.sh $DATASET_NAME $MODEL_NAME $METRICS > ${PBS_JOBID}.log 2>&1
    "
    
    
    