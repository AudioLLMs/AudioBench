

# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# These environment variables are required for our server to run
# You can remove them on your local server
echo "Setting up environment variables"
echo "Warning: These environment variables are required for our server to run. You can remove them on your local server"
export no_proxy=localhost,127.0.0.1,10.104.0.0/21
export https_proxy=http://10.104.4.124:10104
export http_proxy=http://10.104.4.124:10104
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/cache/huggingface_cache
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =


export CUDA_VISIBLE_DEVICES=0
port=5001

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
        
echo "Started server on port $port"

