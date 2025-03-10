import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from app.show_examples import *
from app.content import *

import pandas as pd

from model_information import get_dataframe
info_df = get_dataframe()


def draw(folder_name, category_name, displayname, metrics, cus_sort=True):
    
    folder = f"./results_organized/{metrics}/"

    # Load the results from CSV
    data_path = f'{folder}/{category_name.lower()}.csv'
    chart_data = pd.read_csv(data_path).round(3)
    
    dataset_name = displayname2datasetname[displayname]
    chart_data = chart_data[['Model', dataset_name]]

    # Rename to proper display name
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
    
    chart_data = chart_data[chart_data['model_show'].isin(models)]
    chart_data = chart_data.sort_values(by=[displayname], ascending=cus_sort).dropna(axis=0)

    if len(chart_data) == 0: return



    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    '''
    Show Table
    '''
    with st.container():
        st.markdown('##### TABLE')

        
        model_link = {key.strip(): val for key, val in zip(info_df['Proper Display Name'], info_df['Link'])}

        chart_data['model_link'] = chart_data['model_show'].map(model_link) 

        chart_data_table = chart_data[['model_show', chart_data.columns[1], chart_data.columns[3]]]

        # Format numeric columns to 2 decimal places
        #chart_data_table[chart_data_table.columns[1]] = chart_data_table[chart_data_table.columns[1]].apply(lambda x: round(float(x), 3) if isinstance(float(x), (int, float)) else float(x))
        cur_dataset_name = chart_data_table.columns[1]


        def highlight_first_element(x):
                # Create a DataFrame with the same shape as the input
                df_style = pd.DataFrame('', index=x.index, columns=x.columns)
                # Apply background color to the first element in row 0 (df[0][0])
                # df_style.iloc[0, 1] = 'background-color: #b0c1d7; color: white'
                df_style.iloc[0, 1] = 'background-color: #b0c1d7'
                
                return df_style

        if cur_dataset_name in [
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
                    by=chart_data_table.columns[1],
                    ascending=True
                ).reset_index(drop=True)
        else:
            chart_data_table = chart_data_table.sort_values(
                    by=chart_data_table.columns[1],
                    ascending=False
                ).reset_index(drop=True)
            

        styled_df = chart_data_table.style.format(
            {chart_data_table.columns[1]: "{:.3f}"}
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
            data_values = chart_data.iloc[:, 1]
            
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
                        "data":  chart_data['model_show'].tolist(),
                    }
                ],
                "yAxis": [{"type": "value", 
                            "min": min_value,
                            "max": max_value, 
                            "boundaryGap": True
                            # "splitNumber": 10
                            }],
                "series": [{
                        "name": f"{dataset_name}",
                        "type": "bar",
                        "data": chart_data[f'{displayname}'].tolist(),
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
        
