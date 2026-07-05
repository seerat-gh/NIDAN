from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

raw_data_dir = project_root / "data" / "raw"
processed_data_dir = project_root / "data" / "processed"

x_train_path = raw_data_dir / "X_train_2025.csv"
y_train_path = raw_data_dir / "y_train_2025.csv"

model_dir = project_root / "models"
report_dir = project_root / "reports"
image_dir = project_root / "images"

random_state = 42
test_size = 0.2
cv_folds = 5