import streamlit as st
from app.draw_diagram import draw_table
from app.content import *
from app.summarization import sum_table_mulit_metrix

def dataset_contents(dataset, metrics):
    custom_css = """
                <style>
                .my-dataset-info {
                # background-color: #F9EBEA;
                # padding: 10px;
                color: #050505;
                font-style: normal;
                font-size: 8px;
                height: auto;
                }
                </style>
                """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(f"""<div class="my-dataset-info">
                    <p><b>About this dataset</b>: {dataset}</p>
                    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="my-dataset-info">
                    <p><b>About this metric</b>: {metrics}</p>
                    </div>""", unsafe_allow_html=True)


def dashboard():

    with st.container():
        st.title("Leaderboard for AudioBench")
   
        st.markdown("""
            [gh1]: https://github.com/AudioLLMs/AudioBench
            [gh2]: https://github.com/AudioLLMs/AudioBench
            **Toolkit:** [![GitHub Repo stars](https://img.shields.io/github/stars/AudioLLMs/AudioBench?style=social)][gh1] | 
            [**Paper @ NAACL 2025**](https://arxiv.org/abs/2406.16020) | 
            **Resource for AudioLLMs:** [![GitHub Repo stars](https://img.shields.io/github/stars/AudioLLMs/Awesome-Audio-LLM?style=social)][gh2]
            """)

    st.markdown("""
            #### Recent updates
            - **Jan. 2025**: AudioBench is officially accepted to NAACL 2025!
            - **Jan. 2025**: Update the layout.
            - **Dec. 2024**: Added MuChoMusic dataset for Music Understanding - MCQ Questions. From Paper: https://arxiv.org/abs/2408.01337.
            - **Dec. 2024**: Singlish ASR task added! The datasets are available on [HF](https://huggingface.co/datasets/MERaLiON/MNSC).
            - **Dec. 2024**: Updated layout and added support for comparison between models with similar sizes. 1) Reorganized layout for a better user experience. 2) Added performance summary for each task.
            - **Aug. 2024**: Initial leaderboard is now online.
            """)

    st.divider()
    st.markdown("""
                #### Evaluating Audio-based Large Language Models
                
                - AudioBench is a comprehensive evaluation benchmark designed for general instruction-following audio large language models.
                - AudioBench is an evaluation benchmark that we continually improve and maintain.
                
                Below are the initial 26 datasets that are included in AudioBench. We are now exteneded to over 50 datasets and going to extend to more in the future.
                """
                )

    with st.container():        
        st.markdown('''
                ''')
        
        st.markdown("###### :dart: Our Benchmark includes: ")
        cols = st.columns(8)
        cols[0].metric(label="Tasks", value=">10")
        cols[1].metric(label="Datasets", value=">50")
        cols[2].metric(label="Evaluated Models", value=">10")
    
    st.divider()
    with st.container():
        left_co, right_co = st.columns([1, 0.1])

        with left_co:
            st.markdown("""
                        ##### Citations :round_pushpin:
                        ```
                        @article{wang2024audiobench,
                            title={AudioBench: A Universal Benchmark for Audio Large Language Models},
                            author={Wang, Bin and Zou, Xunlong and Lin, Geyu and Sun, Shuo and Liu, Zhuohan and Zhang, Wenyu and Liu, Zhengyuan and Aw, AiTi and Chen, Nancy F},
                            journal={NAACL},
                            year={2025}
                            }
                        ```
                        ```
                        @article{zhang2024mowe,
                            title={MoWE-Audio: Multitask AudioLLMs with Mixture of Weak Encoders},
                            author={Zhang, Wenyu and Sun, Shuo and Wang, Bin and Zou, Xunlong and Liu, Zhuohan and He, Yingxu and Lin, Geyu and Chen, Nancy F and Aw, Ai Ti},
                            journal={ICASSP},
                            year={2025}
                            }
                        ```
                        ```
                        @article{wang2025advancing,
                            title={Advancing Singlish Understanding: Bridging the Gap with Datasets and Multimodal Models},
                            author={Wang, Bin and Zou, Xunlong and Sun, Shuo and Zhang, Wenyu and He, Yingxu and Liu, Zhuohan and Wei, Chengwei and Chen, Nancy F and Aw, AiTi},
                            journal={arXiv preprint arXiv:2501.01034},
                            year={2025}
                            }
                        ```
                        ```
                        @article{he2024meralion,
                            title={MERaLiON-AudioLLM: Technical Report},
                            author={He, Yingxu and Liu, Zhuohan and Sun, Shuo and Wang, Bin and Zhang, Wenyu and Zou, Xunlong and Chen, Nancy F and Aw, Ai Ti},
                            journal={arXiv preprint arXiv:2412.09818},
                            year={2024}
                            }
                        ```
                        """)


def asr_english():
    st.title("Task: Automatic Speech Recognition - English")
    
    sum = ['Overall']
    dataset_list = [
                    'LibriSpeech-Clean', 
                    'LibriSpeech-Other', 
                    'CommonVoice-15-EN', 
                    'Peoples-Speech', 
                    'GigaSpeech-1', 
                    'Earnings-21', 
                    'Earnings-22', 
                    'TED-LIUM-3', 
                    'TED-LIUM-3-LongForm', 
                    ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['WER'])
        metric = metric.lower()
    
    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)


def asr_singlish():
    st.title("Task: Automatic Speech Recognition - Singlish")

    sum = ['Overall']
    dataset_list = [
                    'MNSC-PART1-ASR', 
                    'MNSC-PART2-ASR',
                    'MNSC-PART3-ASR',
                    'MNSC-PART4-ASR',
                    'MNSC-PART5-ASR',
                    'MNSC-PART6-ASR',
                    'SEAME-Dev-Man',
                    'SEAME-Dev-Sge',
                    ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['WER'])
        metric = metric.lower()
    
    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)




def asr_mandarin():
    st.title("Task: Automatic Speech Recognition - Mandarin")

    sum = ['Overall']
    dataset_list = [
                    'AISHELL-ASR-ZH', 
                    ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['WER'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)


    

def speech_translation():
    st.title("Task: Speech Translation")
    
    sum = ['Overall']
    dataset_list = [
                        'CoVoST2-EN-ID', 
                        'CoVoST2-EN-ZH',
                        'CoVoST2-EN-TA', 
                        'CoVoST2-ID-EN', 
                        'CoVoST2-ZH-EN', 
                        'CoVoST2-TA-EN']
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['BLEU'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)




def speech_question_answering_english():
    st.title("Task: Spoken Question Answering - English")
    
    sum = ['Overall']
    dataset_list = [
                    'CN-College-Listen-MCQ',
                    'DREAM-TTS-MCQ',
                    'SLUE-P2-SQA5', 
                    'Public-SG-Speech-QA', 
                    'Spoken-SQuAD',
                     ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)


def speech_question_answering_singlish():
    st.title("Task: Spoken Question Answering - Singlish")
    
    sum = ['Overall']
    dataset_list = [
              'MNSC-PART3-SQA', 
              'MNSC-PART4-SQA',
              'MNSC-PART5-SQA',
              'MNSC-PART6-SQA',
              ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)


def spoken_dialogue_summarization_singlish():
    st.title("Task: Spoken Dialogue Summarization - Singlish")
    
    sum = ['Overall']
    dataset_list = [
              'MNSC-PART3-SDS', 
              'MNSC-PART4-SDS',
              'MNSC-PART5-SDS',
              'MNSC-PART6-SDS',
              ]
    filters_1_list = sum + dataset_list

    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)




def speech_instruction():
    st.title("Task: Speech Instruction")
    
    sum = ['Overall']
    dataset_list = ['OpenHermes-Audio', 
                     'ALPACA-Audio',
                     ]
    filters_1_list = sum + dataset_list
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)



def audio_captioning():
    st.title("Task: Audio Captioning")

    dataset_list = [    'WavCaps', 
                        'AudioCaps',
                        ]
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', dataset_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE', 'METEOR'])
        metric = metric.lower()

    if tab_section:
        dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
        draw_table(tab_section, metric)


def audio_scene_question_answering():
    st.title("Task: Audio Scene Question Answering")

    sum = ['Overall']
    dataset_list = ['Clotho-AQA', 
                    'WavCaps-QA', 
                    'AudioCaps-QA']
    
    filters_1_list = sum + dataset_list

    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)





def accent_recognition():
    st.title("Task: Accent Recognition")

    sum = ['Overall']
    dataset_list = [
        'VoxCeleb-Accent',
        'MNSC-AR-Sentence',
        'MNSC-AR-Dialogue',
        ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)



def gender_recognition():
    st.title("Task: Gender Recognition")
    
    sum = ['Overall']
    dataset_list =  [
                        'VoxCeleb-Gender', 
                        'IEMOCAP-Gender'
                        ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)





def emotion_recognition():
    st.title("Task: Emotion Recognition")

    sum = ['Overall']
    dataset_list = [
                    'IEMOCAP-Emotion', 
                    'MELD-Sentiment', 
                    'MELD-Emotion',
                    ]
    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)




def music_understanding():
    st.title("Task: Music Understanding - MCQ Questions")
    
    sum = ['Overall']

    dataset_list =  ['MuChoMusic',
                      ]

    filters_1_list = sum + dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1: 
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
        metric = metric.lower()

    if tab_section:
        if tab_section in sum:
            sum_table_mulit_metrix(dataset_list, metric)
        else:
            dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
            draw_table(tab_section, metric)







def under_development():
    st.title("Task: Under Development")
    
    dataset_list =  [
                      'CNA',
                      'IDPC',
                      'Parliament',
                      'UKUS-News',
                      'Mediacorp',
                      'IDPC-Short',
                      'Parliament-Short',
                      'UKUS-News-Short',
                      'Mediacorp-Short',
                      'YTB-ASR-Batch1',
                      'YTB-ASR-Batch2',
                      'YTB-SQA-Batch1',
                      'YTB-SDS-Batch1',
                      'YTB-PQA-Batch1',
                      ]

    filters_1_list = dataset_list
    
    space1, space2, _, _ = st.columns([0.4, 0.4, 0.2 ,0.2])
    
    with space1:
        tab_section = st.selectbox('Dataset', filters_1_list)
    with space2:
        if tab_section in [
                'CNA',
                'IDPC',
                'Parliament',
                'UKUS-News',
                'Mediacorp',
                'IDPC-Short',
                'Parliament-Short',
                'UKUS-News-Short',
                'Mediacorp-Short',
                'YTB-ASR-Batch1',
                'YTB-ASR-Batch2',
                ]:
            metric = st.selectbox('Metric', ['WER'])
            metric = metric.lower()
        elif tab_section in [
                'YTB-SQA-Batch1',
                'YTB-SDS-Batch1',
                'YTB-PQA-Batch1',
                ]:
            metric = st.selectbox('Metric', ['LLAMA3_70B_JUDGE'])
            metric = metric.lower()
        else:
            raise ValueError('Invalid dataset')

    
    if tab_section:
        dataset_contents(dataset_diaplay_information[tab_section], metrics_info[metric])
        draw_table(tab_section, metric)


def mmau_evaluation():
    st.title("Task: MMAU-Audio Understanding")


