# Wellness Tourism Package — MLOps Project

End-to-end MLOps pipeline predicting whether a customer will purchase the **Wellness Tourism Package** offered by 'Visit with Us' travel company.

## Project Overview

| Component | Details |
|---|---|
| **Model** | XGBoost with GridSearchCV (16 combinations, 5-fold CV) |
| **ROC-AUC** | 95.28% |
| **F1-Score** | 77.14% |
| **Dataset** | 4,128 customers x 18 features |

## Repository Structure

    tourism-mlops-project/
    .github/workflows/pipeline.yml   # GitHub Actions CI/CD (4 jobs)
    tourism_project/
        model_building/
            data_register.py         # Upload raw data to HF Hub
            prep.py                  # Clean, split, re-upload
            train.py                 # Train XGBoost, log MLflow
        deployment/
            app.py                   # Streamlit web app
            Dockerfile
            requirements.txt
        hosting/
            hosting.py               # Push app to HF Spaces
    tourism.csv

## GitHub Actions Pipeline

4 sequential jobs triggered on every push to main:

1. **register-dataset** — Uploads raw CSV to HuggingFace dataset hub
2. **data-prep** — Cleans data, fixes anomalies, splits 80/20
3. **model-training** — Trains XGBoost, tracks with MLflow, registers model
4. **deploy-hosting** — Pushes Streamlit app to HF Spaces

## Links

- **Live App:** https://huggingface.co/spaces/t2arin2003/tourism-predictor-app
- **Dataset:** https://huggingface.co/datasets/t2arin2003/tourism-package-dataset
- **Model:** https://huggingface.co/t2arin2003/tourism-package-model

## Tech Stack

Python | XGBoost | scikit-learn | MLflow | Streamlit | HuggingFace Hub | Docker | GitHub Actions
