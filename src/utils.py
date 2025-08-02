import pandas as pd

def load_data():
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    y = train["target"]
    X = train.drop(columns=["target", "id"])
    X_test = test.drop(columns=["id"])
    return X, y, X_test
