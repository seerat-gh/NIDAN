import joblib
import pandas as pd

from config import model_dir


def load_model():
    return joblib.load(model_dir / "best_model.pkl")


def predict_single(sample):
    model = load_model()

    sample = pd.DataFrame([sample])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    return prediction, probability


def predict_csv(file_path):
    model = load_model()

    data = pd.read_csv(file_path)

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    result = data.copy()
    result["prediction"] = predictions
    result["probability"] = probabilities

    return result


if __name__ == "__main__":

    print("predict.py loaded successfully.")