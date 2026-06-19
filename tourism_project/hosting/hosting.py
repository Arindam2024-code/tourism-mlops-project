"""
Hosting Script — Push Streamlit app to Hugging Face Spaces.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_USER       = os.environ.get("HF_USERNAME", "your-hf-username")
SPACE_REPO_ID = f"{HF_USER}/tourism-package-predictor"
DEPLOY_DIR    = "tourism_project/deployment"

api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
    print(f"Space '{SPACE_REPO_ID}' exists.")
except RepositoryNotFoundError:
    print(f"Creating space '{SPACE_REPO_ID}' ...")
    create_repo(repo_id=SPACE_REPO_ID, repo_type="space",
                space_sdk="docker", private=False, token=HF_TOKEN)

print(f"Uploading '{DEPLOY_DIR}' to {SPACE_REPO_ID} ...")
api.upload_folder(folder_path=DEPLOY_DIR, repo_id=SPACE_REPO_ID,
                  repo_type="space", token=HF_TOKEN)
print(f"Deployment complete: https://huggingface.co/spaces/{SPACE_REPO_ID}")
