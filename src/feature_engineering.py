from data_loader import load_data
from preprocessing import preprocess_data


def feature_engineering(x):
    """Perform feature engineering."""

    print("=" * 60)
    print("feature engineering")
    print("=" * 60)

    # add feature engineering steps here

    return x


if __name__ == "__main__":
    x, y = load_data()

    x, y = preprocess_data(x, y)

    x = feature_engineering(x)

    print("\nfeature engineering completed successfully.")
    print(f"feature shape : {x.shape}")