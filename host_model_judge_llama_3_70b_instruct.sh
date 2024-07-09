



export CUDA_VISIBLE_DEVICES=0,1

port=5000

python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3-70B-Instruct \
        --port $port \
        --tensor-parallel-size 2 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
        
echo "Started server on port $port"

