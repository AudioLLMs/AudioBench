import streamlit as st
from app.draw_diagram import *
from app.content import *
from app.summarization import *

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
                
                Below are the initial 26 datasets that are included in AudioBench. We are now exteneded to over 40 datasets and going to extend to more in the future.
                """
                )


    with st.container():
        
        st.markdown('''
                ''')
        
        st.markdown("###### :dart: Our Benchmark includes: ")
        cols = st.columns(8)
        cols[0].metric(label="Tasks", value=">8")
        cols[1].metric(label="Datasets", value=">40")
        cols[2].metric(label="Evaluated Models", value=">5")
    
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
    dataset_lists = [
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

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('asr_english', ['wer'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['wer'])
            draw('su', 'asr_english', filter_1, 'wer', cus_sort=True)





def asr_singlish():
    st.title("Task: Automatic Speech Recognition - Singlish")

    sum = ['Overall']
    dataset_lists = [
                    'MNSC-PART1-ASR', 
                    'MNSC-PART2-ASR',
                    'MNSC-PART3-ASR',
                    'MNSC-PART4-ASR',
                    'MNSC-PART5-ASR',
                    'MNSC-PART6-ASR',
                    'SEAME-Dev-Man',
                    'SEAME-Dev-Sge',
                    ]

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('asr_singlish', ['wer'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['wer'])
            draw('su', 'asr_singlish', filter_1, 'wer')




def asr_mandarin():
    st.title("Task: Automatic Speech Recognition - Mandarin")

    sum = ['Overall']
    dataset_lists = [
                    'AISHELL-ASR-ZH', 
                    ]

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('asr_mandarin', ['wer'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['wer'])
            draw('su', 'asr_mandarin', filter_1, 'wer')

    


def speech_translation():
    st.title("Task: Speech Translation")
    
    sum = ['Overall']
    dataset_lists = [
                        'CoVoST2-EN-ID', 
                        'CoVoST2-EN-ZH',
                        'CoVoST2-EN-TA', 
                        'CoVoST2-ID-EN', 
                        'CoVoST2-ZH-EN', 
                        'CoVoST2-TA-EN']

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('st', ['bleu'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['bleu'])
            draw('su', 'ST', filter_1, 'bleu')




def speech_question_answering_english():
    st.title("Task: Spoken Question Answering - English")
    
    sum = ['Overall']

    dataset_lists = [
                    'CN-College-Listen-MCQ',
                    'DREAM-TTS-MCQ',
                    'SLUE-P2-SQA5', 
                    'Public-SG-Speech-QA', 
                    'Spoken-SQuAD',
                     ]

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)

    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('sqa_english', ['llama3_70b_judge'])

        #elif filter_1 in dataset_lists:
        #    dataset_contents(sqa_datasets[filter_1], metrics['llama3_70b_judge'])
        #    draw('su', 'SQA', filter_1, 'llama3_70b_judge')
        
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('su', 'sqa_english', filter_1, 'llama3_70b_judge')




def speech_question_answering_singlish():
    st.title("Task: Spoken Question Answering - Singlish")
    
    sum = ['Overall']

    dataset_lists = [
              'MNSC-PART3-SQA', 
              'MNSC-PART4-SQA',
              'MNSC-PART5-SQA',
              'MNSC-PART6-SQA',
              ]


    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left: 
        filter_1 = st.selectbox('Dataset', filters_levelone)

    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('sqa_singlish', ['llama3_70b_judge'])
        
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('su', 'sqa_singlish', filter_1, 'llama3_70b_judge')


def spoken_dialogue_summarization_singlish():
    st.title("Task: Spoken Dialogue Summarization - Singlish")
    
    sum = ['Overall']

    dataset_lists = [
              'MNSC-PART3-SDS', 
              'MNSC-PART4-SDS',
              'MNSC-PART5-SDS',
              'MNSC-PART6-SDS',
              ]


    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left: 
        filter_1 = st.selectbox('Dataset', filters_levelone)

    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('sds_singlish', ['llama3_70b_judge'])
        
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('su', 'sds_singlish', filter_1, 'llama3_70b_judge')




def speech_instruction():
    st.title("Task: Speech Instruction")
    
    sum = ['Overall']

    dataset_lists = ['OpenHermes-Audio', 
                     'ALPACA-Audio',
                     ]
    
    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)

    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('speech_instruction', ['llama3_70b_judge'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('su', 'speech_instruction', filter_1, 'llama3_70b_judge')




def audio_captioning():
    st.title("Task: Audio Captioning")

    filters_levelone = ['WavCaps', 
                        'AudioCaps',
                        ]
    filters_leveltwo = ['Llama3-70b-judge', 'Meteor']
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    with middle:
        metric = st.selectbox('Metric', filters_leveltwo)

    if filter_1 or metric:
        dataset_contents(dataset_diaplay_information[filter_1], metrics_info[metric.lower().replace('-', '_')])
        draw('asu', 'audio_captioning', filter_1, metric.lower().replace('-', '_'))




def audio_scene_question_answering():
    st.title("Task: Audio Scene Question Answering")

    sum = ['Overall']

    dataset_lists = ['Clotho-AQA', 
                    'WavCaps-QA', 
                    'AudioCaps-QA']
    
    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('audio_scene_question_answering', ['llama3_70b_judge'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('asu', 'audio_scene_question_answering', filter_1, 'llama3_70b_judge')




def emotion_recognition():
    st.title("Task: Emotion Recognition")

    sum = ['Overall']

    dataset_lists = [
                    'IEMOCAP-Emotion', 
                    'MELD-Sentiment', 
                    'MELD-Emotion',
                    ]

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)

    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('emotion_recognition', ['llama3_70b_judge'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('vu', 'emotion_recognition', filter_1, 'llama3_70b_judge')




def accent_recognition():
    st.title("Task: Accent Recognition")

    sum = ['Overall']
    dataset_lists = [
        'VoxCeleb-Accent',
        'MNSC-AR-Sentence',
        'MNSC-AR-Dialogue',
        ]


    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)


    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('accent_recognition', ['llama3_70b_judge'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('vu', 'accent_recognition', filter_1, 'llama3_70b_judge')




def gender_recognition():
    st.title("Task: Gender Recognition")
    
    sum = ['Overall']

    dataset_lists =  [
                        'VoxCeleb-Gender', 
                        'IEMOCAP-Gender'
                        ]

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('gender_recognition', ['llama3_70b_judge'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('vu', 'gender_recognition', filter_1, 'llama3_70b_judge')




def music_understanding():
    st.title("Task: Music Understanding - MCQ Questions")
    
    sum = ['Overall']

    dataset_lists =  ['MuChoMusic',
                      ]

    filters_levelone = sum + dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    if filter_1:
        if filter_1 in sum:
            sum_table_mulit_metrix('music_understanding', ['llama3_70b_judge'])
        else:
            dataset_contents(dataset_diaplay_information[filter_1], metrics_info['llama3_70b_judge'])
            draw('vu', 'music_understanding', filter_1, 'llama3_70b_judge')










def under_development():
    st.title("Task: Under Development")
    

    dataset_lists =  [
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

    filters_levelone = dataset_lists
    
    left, center, _, middle, right = st.columns([0.4, 0.2, 0.2, 0.2 ,0.2])
    
    with left:
        filter_1 = st.selectbox('Dataset', filters_levelone)
    
    dataset_contents(dataset_diaplay_information[filter_1], 'under_development')
    
    if filter_1 in [
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
                      'SEAME-Dev-Man',
                      'SEAME-Dev-Sge',
                      ]:
        
        draw('vu', 'under_development_wer', filter_1, 'wer')

    elif filter_1 in [
        'YTB-SQA-Batch1',
        'YTB-SDS-Batch1',
        'YTB-PQA-Batch1',
        ]:
        draw('vu', 'under_development_llama3_70b_judge', filter_1, 'llama3_70b_judge')





