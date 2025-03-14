import streamlit as st
import pandas as pd
import numpy as np

import json

from streamlit_echarts import st_echarts
from streamlit.components.v1 import html
# from PIL import Image 
from app.show_examples import *
from app.content import *

import pandas as pd
from typing import List

from model_information import get_dataframe

info_df = get_dataframe()

def sum_table_mulit_metrix(dataset_displayname_list, metric):

    with open('organize_model_results.json', 'r') as f:
        organize_model_results = json.load(f)

    dataset_results = {}

    for dataset_displayname in dataset_displayname_list:
        dataset_nickname = displayname2datasetname[dataset_displayname]
        model_results = organize_model_results[dataset_nickname][metric]
        model_name_mapping = {key.strip(): val for key, val in zip(info_df['Original Name'], info_df['Proper Display Name'])}
        model_results      = {model_name_mapping.get(key, key): val for key, val in model_results.items()}

        dataset_results[dataset_displayname] = model_results

    df_results = pd.DataFrame(dataset_results)

    # Reset index to have models as a column
    df_results.reset_index(inplace=True)
    df_results.rename(columns={"index": "Model"}, inplace=True)
    chart_data = df_results    

    selected_columns = [i for i in chart_data.columns if i != 'Model']
    chart_data['Average'] = chart_data[selected_columns].mean(axis=1)

    # Update dataset name in table
    chart_data = chart_data.rename(columns=datasetname2diaplayname)
    
    st.markdown("""
                <style>
                .stMultiSelect [data-baseweb=select] span {
                    max-width: 800px;
                    font-size: 0.9rem;
                    background-color: #3C6478 !important; /* Background color for selected items */
                    color: white; /* Change text color */
                    back
                }
                </style>
                """, unsafe_allow_html=True)
    
    # remap model names
    display_model_names = {key.strip() :val.strip() for key, val in zip(info_df['Original Name'], info_df['Proper Display Name'])}
    chart_data['model_show'] = chart_data['Model'].map(lambda x: display_model_names.get(x, x))

    models = st.multiselect("Please choose the model", 
                            sorted(chart_data['model_show'].tolist()), 
                            default = sorted(chart_data['model_show'].tolist()),
                            )
    
    chart_data = chart_data[chart_data['model_show'].isin(models)].dropna(axis=0)

    if len(chart_data) == 0: return

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    '''
    Show Table
    '''
    with st.container():
        st.markdown(f'##### TABLE')

        model_link = {key.strip(): val for key, val in zip(info_df['Proper Display Name'], info_df['Link'])}

        chart_data['model_link'] = chart_data['model_show'].map(model_link) 

        tabel_columns = [i for i in chart_data.columns if i not in ['Model', 'model_show']]
        column_to_front = 'Average'
        new_order = [column_to_front] + [col for col in tabel_columns if col != column_to_front]
        
        chart_data_table = chart_data[['model_show'] + new_order]
        

        # Format numeric columns to 2 decimal places
        chart_data_table[chart_data_table.columns[1]] = chart_data_table[chart_data_table.columns[1]].apply(lambda x: round(float(x), 3) if isinstance(float(x), (int, float)) else float(x))

        if metric == 'wer':
            ascend = True
        else:
            ascend= False

        chart_data_table = chart_data_table.sort_values(
                by=['Average'],
                ascending=ascend
            ).reset_index(drop=True)
        
        # Highlight the best performing model
        def highlight_first_element(x):
            # Create a DataFrame with the same shape as the input
            df_style = pd.DataFrame('', index=x.index, columns=x.columns)
            # Apply background color to the first element in row 0 (df[0][0])
            # df_style.iloc[0, 1] = 'background-color: #b0c1d7; color: white'
            df_style.iloc[0, 1] = 'background-color: #b0c1d7'

            return df_style
        

        styled_df = chart_data_table.style.format(
            {
                chart_data_table.columns[i]: "{:.3f}" for i in range(1, len(chart_data_table.columns) - 1)
             }
        ).apply(
            highlight_first_element, axis=None
        )

        st.dataframe(
                styled_df,
                column_config={
                    'model_show': 'Model',
                    chart_data_table.columns[1]: {'alignment': 'left'},
                    "model_link": st.column_config.LinkColumn(
                        "Model Link",
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
    # Only report the last metrics
    st.markdown(f'###### Metric: {metrics_info[metric]}')
