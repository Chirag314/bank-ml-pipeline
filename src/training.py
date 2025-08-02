import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.features import FeatureGenerator

def train_model(X, y):
    fg = FeatureGenerator()
    fg.fit(X)
    X = fg.transform(X)

    oof_preds = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, preds)}")
        models.append(model)

    return models[0], oof_preds
