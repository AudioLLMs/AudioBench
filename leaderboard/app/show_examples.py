import streamlit as st
import datasets
import numpy as np

import html


def show_examples(category_name, dataset_name, model_lists, display_model_names):
    st.divider()
    sample_folder = f"./examples/{category_name}/{dataset_name}"
    
    dataset = datasets.load_from_disk(sample_folder)

    for index in range(len(dataset)):
        with st.container():
            st.markdown(f'##### Example-{index+1}')
            col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")

            # with col1:
            st.audio(f'{sample_folder}/sample_{index}.wav', format="audio/wav")
                        
            if dataset_name in ['CN-College-Listen-MCQ-Test', 'DREAM-TTS-MCQ-Test']:
                
                choices = dataset[index]['other_attributes']['choices'] 
                if isinstance(choices, str):
                    choices_text = choices
                elif isinstance(choices, list):
                    choices_text = ' '.join(i for i in choices)
                
                question_text = f"""{dataset[index]['instruction']['text']} {choices_text}"""
            else:
                question_text = f"""{dataset[index]['instruction']['text']}"""

            question_text = html.escape(question_text)
            
            # st.divider()
            with st.container():
                custom_css = """
                            <style>
                            .my-container-table, p.my-container-text {
                            background-color: #fcf8dc;
                            padding: 10px;
                            border-radius: 5px;
                            font-size: 13px;
                            # height: 50px;
                            word-wrap: break-word
                            }
                            </style>
                            """
                st.markdown(custom_css, unsafe_allow_html=True)

                model_lists.sort()

                s = f"""<tr>
                       <td><b>REFERENCE</td>
                       <td><b>{html.escape(question_text.replace('(A)', '<br>(A)').replace('(B)', '<br>(B)').replace('(C)', '<br>(C)'))}
                       </td>
                       <td><b>{html.escape(dataset[index]['answer']['text'])}
                       </td>
                </tr>
                """
                if dataset_name in ['CN-College-Listen-MCQ-Test', 'DREAM-TTS-MCQ-Test']:
                    for model in model_lists:
                        try:

                            model_prediction = dataset[index][model]['model_prediction']
                            model_prediction = model_prediction.replace('<','').replace('>','').replace('\n','(newline)').replace('*','')

                            s += f"""<tr>
                                <td>{display_model_names[model]}</td>
                                <td>
                                    {dataset[index][model]['text'].replace('Choices:', '<br>Choices:').replace('(A)', '<br>(A)').replace('(B)', '<br>(B)').replace('(C)', '<br>(C)') 
                                     }
                                    </td>
                                <td>{html.escape(model_prediction)}</td>
                            </tr>"""
                        except:
                            print(f"{model} is not in {dataset_name}")
                            continue
                else:
                    for model in model_lists:

                        print(dataset[index][model]['model_prediction'])

                        try:

                            model_prediction = dataset[index][model]['model_prediction']
                            model_prediction = model_prediction.replace('<','').replace('>','').replace('\n','(newline)').replace('*','')

                            s += f"""<tr>
                                <td>{display_model_names[model]}</td>
                                <td>{html.escape(dataset[index][model]['text'])}</td>
                                <td>{html.escape(model_prediction)}</td>
                            </tr>"""
                        except:
                            print(f"{model} is not in {dataset_name}")
                            continue

                
                body_details = f"""<table style="table-layout: fixed; width:100%">
                <thead>
                    <tr style="text-align: center;">
                        <th style="width:20%">MODEL</th>
                        <th style="width:30%">QUESTION</th>
                        <th style="width:50%">MODEL PREDICTION</th>
                    </tr>
                {s}
                </thead>
                </table>"""
                
                st.markdown(f"""<div class="my-container-table">
                                {body_details}
                                </div>""", unsafe_allow_html=True)
            
                st.text("")
        
        st.divider()

    
    