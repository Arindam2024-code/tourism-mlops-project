"""
Data Registration Script
Uploads the raw tourism dataset to a Hugging Face dataset repository.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_USER   = os.environ.get("HF_USERNAME", "your-hf-username")
REPO_ID   = f"{HF_USER}/tourism-package-dataset"
REPO_TYPE = "dataset"
DATA_DIR  = "tourism_project/data"

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating repo '{REPO_ID}' ...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
    print(f"Repo '{REPO_ID}' created.")

print(f"Uploading '{DATA_DIR}' to {REPO_ID} ...")
api.upload_folder(folder_path=DATA_DIR, repo_id=REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN)
print("Dataset upload complete.")
