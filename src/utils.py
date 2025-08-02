import pandas as pd


def load_data():
    train = pd.read_csv("src/data/train.csv")
    test = pd.read_csv("src/data/test.csv")
    y = train["y"]
    X = train.drop(columns=["y", "id"])
    X_test = test.drop(columns=["id"])
    return X, y, X_test
