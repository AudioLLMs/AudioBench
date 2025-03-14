import streamlit as st
import pandas as pd
import numpy as np
import json

from streamlit_echarts import st_echarts
from app.show_examples import *
from app.content import *

import pandas as pd

from model_information import get_dataframe
info_df = get_dataframe()


def draw_table(dataset_displayname, metrics):

    with open('organize_model_results.json', 'r') as f:
        organize_model_results = json.load(f)

    dataset_nickname   = displayname2datasetname[dataset_displayname]
    model_results      = organize_model_results[dataset_nickname][metrics]
    model_name_mapping = {key.strip(): val for key, val in zip(info_df['Original Name'], info_df['Proper Display Name'])}
    model_results      = {model_name_mapping.get(key, key): val for key, val in model_results.items()}


    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    '''
    Show Table
    '''
    with st.container():
        st.markdown('##### TABLE')

        model_link_mapping             = {key.strip(): val for key, val in zip(info_df['Proper Display Name'], info_df['Link'])}
        chart_data_table               = pd.DataFrame(list(model_results.items()), columns=["model_show", dataset_displayname])
        chart_data_table["model_link"] = chart_data_table["model_show"].map(model_link_mapping)

        def highlight_first_element(x):
                # Create a DataFrame with the same shape as the input
                df_style            = pd.DataFrame('', index=x.index, columns=x.columns)
                df_style.iloc[0, 1] = 'background-color: #b0c1d7'
                return df_style

        if dataset_displayname in [
                            'LibriSpeech-Clean',
                            'LibriSpeech-Other',
                            'CommonVoice-15-EN',
                            'Peoples-Speech',
                            'GigaSpeech-1',
                            'Earnings-21',
                            'Earnings-22',
                            'TED-LIUM-3',
                            'TED-LIUM-3-LongForm',
                            'AISHELL-ASR-ZH',
                            'MNSC-PART1-ASR',
                            'MNSC-PART2-ASR',
                            'MNSC-PART3-ASR',
                            'MNSC-PART4-ASR',
                            'MNSC-PART5-ASR',
                            'MNSC-PART6-ASR',
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
            
            chart_data_table = chart_data_table.sort_values(
                                    by        = chart_data_table.columns[1],
                                    ascending = True
                                ).reset_index(drop=True)
        else:
            chart_data_table = chart_data_table.sort_values(
                                    by        = chart_data_table.columns[1],
                                    ascending = False
                                ).reset_index(drop=True)
                            

        styled_df = chart_data_table.style.format(
                                    {chart_data_table.columns[1]: "{:.3f}"}
                                ).apply(
                                    highlight_first_element, axis=None
                                )


        st.dataframe(
                        styled_df,
                        column_config={
                            'model_show'               : 'Model',
                            chart_data_table.columns[1]: {'alignment': 'left'},
                            "model_link"               : st.column_config.LinkColumn("Model Link"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    '''
    Show Chart
    '''
    # Initialize a session state variable for toggling the chart visibility
    if "show_chart" not in st.session_state:
        st.session_state.show_chart = False

    # Create a button to toggle visibility
    if st.button("Show Chart"):
        st.session_state.show_chart = not st.session_state.show_chart

    if st.session_state.show_chart:

        with st.container():
            st.markdown('##### CHART')

            # Get Values
            data_values = chart_data_table.iloc[:, 1]
            
            # Calculate Q1 and Q3
            q1 = data_values.quantile(0.25)
            q3 = data_values.quantile(0.75)

            # Calculate IQR
            iqr = q3 - q1

            # Define lower and upper bounds (1.5*IQR is a common threshold)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Filter data within the bounds
            filtered_data = data_values[(data_values >= lower_bound) & (data_values <= upper_bound)]

            # Calculate min and max values after outlier handling
            min_value = round(filtered_data.min() - 0.1 * filtered_data.min(), 3)
            max_value = round(filtered_data.max() + 0.1 * filtered_data.max(), 3)

            options = {
                # "title": {"text": f"{dataset_name}"},
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},
                    "triggerOn": 'mousemove',
                },
                "legend": {"data": ['Overall Accuracy']},
                "toolbox": {"feature": {"saveAsImage": {}}},
                "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
                "xAxis": [
                    {
                        "type": "category",
                        "boundaryGap": True,
                        "triggerEvent": True,
                        "data":  chart_data_table['model_show'].tolist(),
                    }
                ],
                "yAxis": [{"type": "value", 
                            "min": min_value,
                            "max": max_value, 
                            "boundaryGap": True
                            # "splitNumber": 10
                            }],
                "series": [{
                        "name": f"{dataset_nickname}",
                        "type": "bar",
                        "data": chart_data_table[f'{dataset_displayname}'].tolist(),
                    }],
            }
            
            events = {
                "click": "function(params) { return params.value }"
            }

            value = st_echarts(options=options, events=events, height="500px")
            

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    '''
    Show Examples
    '''
    # Initialize a session state variable for toggling the chart visibility
    if "show_examples" not in st.session_state:
        st.session_state.show_examples = False

    # Create a button to toggle visibility
    if st.button("Show Examples"):
        st.session_state.show_examples = not st.session_state.show_examples

    if st.session_state.show_examples:
        st.markdown('To be implemented')

        # # if dataset_name in ['Earnings21-Test', 'Earnings22-Test', 'Tedlium3-Test', 'Tedlium3-Long-form-Test']:
        # if dataset_name in []:
        #     pass
        # else:
        #     show_examples(category_name, dataset_name, chart_data['Model'].tolist(), display_model_names)
        
