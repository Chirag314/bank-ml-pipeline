from src.training import train_model
from src.inference import run_inference
from src.utils import load_data

if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model, oof_preds = train_model(X_train, y_train)
    run_inference(model, X_test)
