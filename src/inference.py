import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from kaggle.api.kaggle_api_extended import KaggleApi

def run_inference(model, X_test):
    preds = model.predict_proba(X_test)[:, 1]
    submission = pd.read_csv("input/sample_submission.csv")
    submission["target"] = preds
    submission.to_csv("submission.csv", index=False)

    # Submit to Kaggle
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(file_name="submission.csv", message="Auto submission", competition="your-competition-name")
