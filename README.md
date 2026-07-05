# 🏥 NIDAN

## In-Hospital Mortality Prediction using Machine Learning

NIDAN is an end-to-end Machine Learning application that predicts the risk of in-hospital mortality using patient clinical data.

The project covers the complete ML workflow from data preprocessing to deployment using Streamlit.

---

## Features

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Multiple Machine Learning Models
- Automatic Best Model Selection
- Patient Mortality Prediction
- Interactive Streamlit Dashboard

---

## Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost

The best model was selected using the ROC-AUC score.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib

---

## Project Structure

```
NIDAN/
│
├── app/
│   ├── app.py
│   └── pages/
│       ├── Prediction.py
│       ├── Model_Performance.py
│       └── About.py
│
├── data/
│
├── images/
│
├── models/
│   ├── best_model.pkl
│   └── logistic_regression.pkl
│
├── reports/
│   └── model_comparison.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Go to the project folder

```bash
cd NIDAN
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## Workflow

1. Load Dataset
2. Perform Exploratory Data Analysis
3. Preprocess Data
4. Train Multiple Models
5. Compare Model Performance
6. Save the Best Model
7. Predict Patient Mortality
8. Deploy with Streamlit

---

## Web Application

The Streamlit application includes:

- 🏠 Home
- 🔍 Patient Prediction
- 📊 Model Performance
- ℹ️ About

Users can upload patient data, generate predictions, view model performance, and download prediction results.

---

## Future Improvements

- Hyperparameter Optimization
- Explainable AI (SHAP/LIME)
- Deep Learning Models
- REST API Integration
- Cloud Deployment

---

## Author

**Seerat**