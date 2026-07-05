import matplotlib.pyplot as plt
from data_loader import load_data
from config import image_dir


def dataset_info(x):
    print("\nDataset Info")
    print("-" * 50)
    print(x.info())


def summary_statistics(x):
    print("\nSummary Statistics")
    print("-" * 50)
    print(x.describe())


def target_distribution(y):
    plt.figure(figsize=(6, 5))
    y.iloc[:, 0].value_counts().sort_index().plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("In-hospital Death")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(image_dir / "target_distribution.png")
    plt.close()


def missing_values(x):
    missing = x.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        plt.figure(figsize=(12, 5))
        missing.plot(kind="bar")
        plt.title("Missing Values")
        plt.tight_layout()
        plt.savefig(image_dir / "missing_values.png")
        plt.close()
    else:
        print("\nNo missing values found.")


def correlation_matrix(x):
    numeric = x.select_dtypes(include="number")

    plt.figure(figsize=(12, 10))
    plt.imshow(numeric.corr(), aspect="auto")
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(image_dir / "correlation_matrix.png")
    plt.close()


def main():
    x, y = load_data()

    dataset_info(x)
    summary_statistics(x)

    target_distribution(y)
    missing_values(x)
    correlation_matrix(x)

    print("\nEDA completed successfully.")
    print("Plots saved in images folder.")


if __name__ == "__main__":
    main()