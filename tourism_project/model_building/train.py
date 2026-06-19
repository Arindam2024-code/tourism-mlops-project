"""
Model Training & Registration Script
Trains XGBoost with GridSearchCV, tracks in MLflow, registers on Hugging Face.
"""

import os, joblib
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_USER         = os.environ.get("HF_USERNAME", "your-hf-username")
DATA_REPO_ID    = f"{HF_USER}/tourism-package-dataset"
MODEL_REPO_ID   = f"{HF_USER}/tourism-package-model"
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Tourism-Package-Prediction-Experiment"
MODEL_DIR       = "tourism_project/model_building"
MODEL_PATH      = f"{MODEL_DIR}/tourism_model.pkl"
RANDOM_STATE    = 42

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow URI: {MLFLOW_URI}")

# Load data from Hugging Face
DOWNLOAD_DIR = "hf_download_cache" # Temporary local directory for downloaded files
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Downloading data splits from Hugging Face ...")
X_train_path = hf_hub_download(repo_id=DATA_REPO_ID, filename="Xtrain.csv", repo_type="dataset", local_dir=DOWNLOAD_DIR, token=HF_TOKEN)
X_test_path  = hf_hub_download(repo_id=DATA_REPO_ID, filename="Xtest.csv",  repo_type="dataset", local_dir=DOWNLOAD_DIR, token=HF_TOKEN)
y_train_path = hf_hub_download(repo_id=DATA_REPO_ID, filename="ytrain.csv", repo_type="dataset", local_dir=DOWNLOAD_DIR, token=HF_TOKEN)
y_test_path  = hf_hub_download(repo_id=DATA_REPO_ID, filename="ytest.csv",  repo_type="dataset", local_dir=DOWNLOAD_DIR, token=HF_TOKEN)

# Read from local paths
X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).squeeze()
y_test  = pd.read_csv(y_test_path).squeeze()

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

num_features = X_train.select_dtypes(include="number").columns.tolist()
cat_features = X_train.select_dtypes(include="object").columns.tolist()

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(),                                          num_features),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
)

# Base model
base_model = XGBClassifier(
    use_label_encoder=False, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1
)

# Pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", base_model)])

# Hyperparameter grid
param_grid = {
    "classifier__n_estimators":  [100, 200],
    "classifier__max_depth":     [3, 5],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample":     [0.8, 1.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

with mlflow.start_run(run_name="XGBoost_GridSearchCV"):
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                               scoring="f1", n_jobs=-1, verbose=1, refit=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best params
    for param, val in grid_search.best_params_.items():
        mlflow.log_param(param.replace("classifier__", ""), val)
    print("Best params:", grid_search.best_params_)

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)
    y_prob_test  = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "train_accuracy" : accuracy_score(y_train, y_pred_train),
        "test_accuracy"  : accuracy_score(y_test,  y_pred_test),
        "test_precision" : precision_score(y_test, y_pred_test, zero_division=0),
        "test_recall"    : recall_score(y_test, y_pred_test, zero_division=0),
        "test_f1"        : f1_score(y_test, y_pred_test, zero_division=0),
        "test_roc_auc"   : roc_auc_score(y_test, y_prob_test),
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, round(v, 4))
        print(f"  {k}: {v:.4f}")

    # Save and log model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.sklearn.log_model(best_model, artifact_path="tourism_xgboost_model")
    mlflow.log_artifact(MODEL_PATH, artifact_path="joblib_model")
    print(f"Model saved: {MODEL_PATH}")
    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

# Upload to HF model hub
api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, token=HF_TOKEN)

api.upload_folder(folder_path=MODEL_DIR, repo_id=MODEL_REPO_ID,
                  repo_type="model", token=HF_TOKEN)
print(f"Model uploaded to Hugging Face Model Hub: {MODEL_REPO_ID}")
