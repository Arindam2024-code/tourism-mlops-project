"""
Data Preparation Script
Loads raw data from HF, cleans it, splits into train/test,
and uploads split CSVs back to the HF dataset repo.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo, delete_file # Import delete_file
from huggingface_hub.utils import RepositoryNotFoundError
import datetime # Import datetime for unique commit messages
import shutil # Import shutil for directory operations

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_USER      = os.environ.get("HF_USERNAME", "your-hf-username")
REPO_ID      = f"{HF_USER}/tourism-package-dataset"
REPO_TYPE    = "dataset"
DATASET_PATH = f"hf://datasets/{REPO_ID}/tourism.csv"
OUTPUT_DIR   = "tourism_project/data"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Ensure output directory is clean before saving new files to force upload
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load from Hugging Face
print("Loading dataset from Hugging Face ...")
df = pd.read_csv(DATASET_PATH, storage_options={"token": HF_TOKEN})
print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns.")

# Drop unnecessary columns
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped: {drop_cols}")

# Fix Gender anomaly
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Fix MaritalStatus anomaly
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Impute nulls (none expected, but safe practice)
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(f"  Nulls remaining: {df.isnull().sum().sum()}")

# Separate target and features
y = df[["ProdTaken"]]
X = df.drop(columns=["ProdTaken"])

# Train/test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Save locally
X_train.to_csv(f"{OUTPUT_DIR}/Xtrain.csv", index=False)
X_test.to_csv( f"{OUTPUT_DIR}/Xtest.csv",  index=False)
y_train.to_csv(f"{OUTPUT_DIR}/ytrain.csv", index=False)
y_test.to_csv( f"{OUTPUT_DIR}/ytest.csv",  index=False)
print(f"  Saved split files to '{OUTPUT_DIR}/'")

# Verify local file sizes
print("  Verifying local file sizes before upload:")
for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    local_file_path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(local_file_path):
        print(f"    {file_name}: {os.path.getsize(local_file_path)} bytes")
    else:
        print(f"    {file_name}: NOT FOUND LOCALLY")


# Upload back to Hugging Face
api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)

# Delete existing data splits from Hugging Face to force a fresh upload
print("  Deleting existing data splits from Hugging Face (if any) ...")
for filename in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    try:
        delete_file(path_in_repo=filename, repo_id=REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN)
        print(f"    Deleted {filename}")
    except Exception as e:
        # File might not exist, which is fine.
        print(f"    Skipping delete for {filename}: {e}")

# Force a new commit message to ensure upload even if content hashes are the same
commit_message = f"Upload data splits - {datetime.datetime.now().isoformat()[:-7]}"

print(f"Uploading '{OUTPUT_DIR}' to {REPO_ID} with commit message: {commit_message} ...")
api.upload_folder(folder_path=OUTPUT_DIR, repo_id=REPO_ID, repo_type=REPO_TYPE, token=HF_TOKEN, commit_message=commit_message)
print("Data preparation complete — splits uploaded to Hugging Face.")
