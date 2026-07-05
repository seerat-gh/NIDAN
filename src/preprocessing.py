from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import random_state, test_size
from data_loader import load_data


def preprocess_data():
    x, y = load_data()

    y = y.squeeze()

    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x.select_dtypes(include=["object", "category", "bool"]).columns

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data()

    print(f"x_train : {x_train.shape}")
    print(f"x_test  : {x_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")