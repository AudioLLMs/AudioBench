
export no_proxy=localhost,127.0.0.1,10.104.0.0/21
export https_proxy=http://10.104.4.124:10104
export http_proxy=http://10.104.4.124:10104



DATASET=$1
MODEL=$2
GPU=$3
BATCH_SIZE=$4
OVERWRITE=$5
METRICS=$6
NUMBER_OF_SAMPLES=$7


export CUDA_VISIBLE_DEVICES=$GPU




python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES
