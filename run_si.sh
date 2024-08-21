export AZURE_OPENAI_KEY="7e703cc81fc8436dbe5045e5bb81b5f0"

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_NAME=llama3.1-s-whisperspeech
GPU=2
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=50
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# SI
DATASET=openhermes_audio_test
DATASET=alpaca_audio_test

METRICS=gpt4_judge


bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES