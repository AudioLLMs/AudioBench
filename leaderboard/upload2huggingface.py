
import os
from huggingface_hub import HfApi, login

# Retrieve token from environment variable
hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please export it in your shell.")

# Log in using the token
login(token=hf_token)
repo_id = "AudioLLMs/AudioBench-Leaderboard"

api = HfApi()
api.upload_folder(repo_id=repo_id, 
                  repo_type="space", 
                  folder_path="./results_organized",
                  path_in_repo="results_organized/")

# api.delete_folder(repo_id=repo_id, 
#                   repo_type="space", 
#                   path_in_repo="examples")

# api.upload_folder(repo_id=repo_id, 
#                   repo_type="space", 
#                   folder_path="./examples",
#                   path_in_repo="examples/")

api.upload_folder(repo_id=repo_id, 
                  repo_type="space", 
                  folder_path="./app",
                  path_in_repo="app/")

api.upload_file(repo_id=repo_id, 
                  repo_type="space", 
                  path_or_fileobj="./requirements.txt",
                  path_in_repo = 'requirements.txt')

api.upload_file(repo_id=repo_id, 
                  repo_type="space", 
                  path_or_fileobj="./app.py",
                  path_in_repo = 'app.py')

api.upload_file(repo_id=repo_id, 
                  repo_type="space", 
                  path_or_fileobj="./model_information.py",
                  path_in_repo = 'model_information.py')

api.restart_space(repo_id=repo_id)
