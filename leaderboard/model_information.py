import pandas as pd

# Define the data
data = {
    "Original Name"      : [],
    "Proper Display Name": [],
    "Link"               : [],
}

# Add model information to the
data['Original Name'].append('SALMONN_7B')
data['Proper Display Name'].append('Fusion: SALMONN-7B')
data['Link'].append('https://arxiv.org/html/2310.13289v2')

data['Original Name'].append('WavLLM_fairseq')
data['Proper Display Name'].append('Fusion: WavLLM')
data['Link'].append('https://arxiv.org/abs/2404.00656')

data['Original Name'].append('Qwen2-Audio-7B-Instruct')
data['Proper Display Name'].append('Fusion: Qwen2-Audio-7B-Instruct')
data['Link'].append('https://arxiv.org/abs/2407.10759')

data['Original Name'].append('cascade_whisper_large_v3_llama_3_8b_instruct')
data['Proper Display Name'].append('Cascade: Whisper-Large-v3 / Llama-3-8B-Instruct')
data['Link'].append('https://arxiv.org/abs/2406.16020')

data['Original Name'].append('mowe_audio')
data['Proper Display Name'].append('Fusion: MOWE-Audio')
data['Link'].append('https://arxiv.org/abs/2409.06635')

data['Original Name'].append('Qwen-Audio-Chat')
data['Proper Display Name'].append('Fusion: Qwen-Audio-Chat')
data['Link'].append('https://arxiv.org/abs/2311.07919')

data['Original Name'].append('MERaLiON-AudioLLM-Whisper-SEA-LION')
data['Proper Display Name'].append('Fusion: MERaLiON-AudioLLM-Whisper-SEA-LION')
data['Link'].append('https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION')

data['Original Name'].append('cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct')
data['Proper Display Name'].append('Cascade: Whisper-Large-v2 / SEA-LIONv3')
data['Link'].append('https://github.com/aisingapore/sealion')

data['Original Name'].append('whisper_large_v3')
data['Proper Display Name'].append('Whisper-large-v3')
data['Link'].append('https://huggingface.co/openai/whisper-large-v3')

data['Original Name'].append('gemini-1.5-flash')
data['Proper Display Name'].append('Gemini-1.5-Flash')
data['Link'].append('https://ai.google.dev/gemini-api/docs/models/gemini')

data['Original Name'].append('phi_4_multimodal_instruct')
data['Proper Display Name'].append('Phi-4-Multimodal-Instruct')
data['Link'].append('https://huggingface.co/microsoft/Phi-4-multimodal-instruct')

data['Original Name'].append('seallms_audio_7b')
data['Proper Display Name'].append('SeaLLMs-Audio-7B')
data['Link'].append('https://huggingface.co/SeaLLMs/SeaLLMs-Audio-7B')

data['Original Name'].append('Marco-LLM-ST')
data['Proper Display Name'].append('Marco-LLM-ST')
data['Link'].append('https://arxiv.org/abs/2412.04003')



def get_dataframe():
    """
    Returns a DataFrame with the data and drops rows with missing values.
    """
    df = pd.DataFrame(data)
    return df.dropna(axis=0)


