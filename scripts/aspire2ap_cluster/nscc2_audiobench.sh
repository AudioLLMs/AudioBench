

export CUDA_VISIBLE_DEVICES=0

# This is required for aspire2a+ cluster
export no_proxy=localhost,127.0.0.1,10.104.0.0/21
export https_proxy=http://10.104.4.124:10104
export http_proxy=http://10.104.4.124:10104


echo "The path to HF ENDPOINT is: $HF_ENDPOINT"
echo "The path to HF HOME is: $HF_HOME"


##### 
GPU=1
BATCH_SIZE=1
OVERWRITE=False
OVERWRITE=True
NUMBER_OF_SAMPLES=-1
#####

DATASET=$1
MODEL_NAME=$2
METRICS=$3

echo "DATASET: $DATASET"
echo "MODEL_NAME: $MODEL_NAME"
echo "METRICS: $METRICS"

bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES