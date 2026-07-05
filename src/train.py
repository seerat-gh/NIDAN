import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier

from preprocessing import preprocess_data
from config import model_dir, random_state, cv_folds


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
        ),
    }


def train_models():
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data()

    models = get_models()

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = []
    best_model = None
    best_score = 0

    for name, model in models.items():

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )

        pipeline.fit(x_train, y_train)

        roc_auc = scores["test_roc_auc"].mean()

        results.append(
            {
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1 Score": scores["test_f1"].mean(),
                "ROC_AUC": roc_auc,
            }
        )

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = pipeline

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="ROC_AUC", ascending=False)

    model_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(model_dir.parent / "reports" / "model_comparison.csv", index=False)

    joblib.dump(best_model, model_dir / "best_model.pkl")

    print("\nModel Comparison:\n")
    print(results_df)

    print("\nBest Model Saved:")
    print(results_df.iloc[0])


if __name__ == "__main__":
    train_models()