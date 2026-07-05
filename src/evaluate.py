import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

from preprocessing import preprocess_data
from config import model_dir, image_dir


def load_model():
    return joblib.load(model_dir / "best_model.pkl")


def evaluate_model():
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data()

    model = load_model()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nClassification Report:\n")
    print(report)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(image_dir / "confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(image_dir / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.savefig(image_dir / "pr_curve.png")
    plt.close()

    print("\nEvaluation completed. Plots saved in images folder.")


if __name__ == "__main__":
    evaluate_model()