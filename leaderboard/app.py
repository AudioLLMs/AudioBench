import streamlit as st
import streamlit_antd_components as sac

from app.pages import *


# Set page configuration
st.set_page_config(
    page_title="AudioBench Leaderboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)



# Dictionary mapping menu items to their corresponding functions
pages = {
    'Dashboard'          : dashboard,
    'ASR-English'        : asr_english,
    'ASR-Mandarin'       : asr_mandarin,
    'ASR-Singlish'       : asr_singlish,
    'Speech Translation' : speech_translation,
    'SQA-English'        : speech_question_answering_english,
    'SQA-Singlish'       : speech_question_answering_singlish,
    'SDS-Singlish'       : spoken_dialogue_summarization_singlish,
    'Speech Instruction' : speech_instruction,
    'Audio Captioning'   : audio_captioning,
    'Audio-Scene QA'     : audio_scene_question_answering,
    'Accent Recognition' : accent_recognition,
    'Gender Recognition' : gender_recognition,
    'Emotion Recognition': emotion_recognition,
    'Music Understanding': music_understanding,

    '* Under Development *': under_development,
}

# Initialize session state for menu selection
if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = 'Introduction'

# Define the menu items
menu_items = [
    sac.MenuItem(label='Dashboard', icon='house'),

    sac.MenuItem(label='Automatic Speech Recognition', icon='mic',
                 children = [
                     sac.MenuItem(label='ASR-English', icon='mic'),
                     sac.MenuItem(label='ASR-Mandarin', icon='mic'),
                     sac.MenuItem(label='ASR-Singlish', icon='mic'),
                 ]
                 ),

    sac.MenuItem(label='Speech Translation', icon='translate'
                 ),

    sac.MenuItem(label='Spoken Question Answering', icon='question-circle',
                 children = [
                     sac.MenuItem(label='SQA-English', icon='mic'),
                     sac.MenuItem(label='SQA-Singlish', icon='mic'),
                 ]
                 ),

    sac.MenuItem(label='Spoken Dialogue Summarization', icon='question-circle',
                 children = [
                     sac.MenuItem(label='SDS-Singlish', icon='mic'),
                 ]
                 ),

    sac.MenuItem(label='Speech Instruction', icon='mic-fill'),

    sac.MenuItem(label='Audio Captioning', icon='volume-down'),

    sac.MenuItem(label='Audio-Scene QA', icon='question-diamond-fill'),
    
    sac.MenuItem(label='Accent Recognition', icon='person-badge-fill'),
    
    sac.MenuItem(label='Gender Recognition', icon='gender-ambiguous'),

    sac.MenuItem(label='Emotion Recognition', icon='emoji-smile-fill'),

    sac.MenuItem(label='Music Understanding', icon='music-note-list'),

    sac.MenuItem(label='* Under Development *', icon='lock'),

]

# Render the menu in the sidebar
with st.sidebar:
    selected = sac.menu(menu_items,
                        size='sm', 
                        open_all=False,
                        )

# Update session state based on selection
if selected:
    st.session_state.selected_menu = selected

# Display the selected page's content
page = pages[st.session_state.selected_menu]
page()
