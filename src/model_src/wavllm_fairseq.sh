#cd /home/Collaborative_Projects/SpeechEval-Related/prepared_models/wavllm_fairseq
#cd /mnt/home/wang_bin/workspaces/SpeechEval-Related/prepared_models/wavllm_fairseq

#cd /mnt/home/wang_bin/workspaces/SpeechEval-Related/AudioBench_private/examples/wavllm_fairseq

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache

dataset_name=$1

echo "Dataset_name: $dataset_name"

cd /project/AudioBench/examples/wavllm_fairseq

bash examples/wavllm/scripts/inference_sft.sh $dataset_name