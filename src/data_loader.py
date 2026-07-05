import pandas as pd

from config import x_train_path, y_train_path


def load_data():
    x = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path)

    return x, y


def dataset_summary(x, y):
    print("\nDataset Summary")
    print("-" * 50)

    print(f"Features Shape : {x.shape}")
    print(f"Target Shape   : {y.shape}")

    print("\nTarget Column:")
    print(y.columns.tolist())

    print("\nTarget Distribution:")
    print(y.iloc[:, 0].value_counts())

    print("\nMissing Values:")
    print(x.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(x.duplicated().sum())


if __name__ == "__main__":
    x, y = load_data()
    dataset_summary(x, y)