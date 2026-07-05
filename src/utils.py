from pathlib import Path
import joblib


def create_directory(directory):
    """Create a directory if it does not exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_model(model, model_path):
    """Save a trained model."""
    create_directory(Path(model_path).parent)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def load_model(model_path):
    """Load a saved model."""
    return joblib.load(model_path)