import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureGenerator:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X.select_dtypes(include='number').fillna(0))

    def transform(self, X):
        X_scaled = self.scaler.transform(X.select_dtypes(include='number').fillna(0))
        return pd.DataFrame(X_scaled, columns=X.select_dtypes(include='number').columns)
