#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com


# rm -r results_organized
python organize_model_results.py

# This will create examples for display. Can be heavy, so skip for now.
# python create_examples.py

# For test how the website looks like
# streamlit run app.py
python upload2huggingface.py
