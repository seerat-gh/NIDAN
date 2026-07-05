import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ About NIDAN")

st.write(
    """
NIDAN is an end-to-end machine learning project developed to predict
the risk of in-hospital mortality using clinical patient data.

The objective is to assist healthcare professionals by identifying
patients who may require immediate medical attention.
"""
)

st.divider()

st.subheader("🎯 Project Workflow")

st.markdown("""
1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Comparison
7. Prediction using Best Model
8. Streamlit Web Application
""")

st.divider()

st.subheader("🤖 Machine Learning Models")

st.markdown("""
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
""")

st.divider()

st.subheader("🛠️ Technologies Used")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- Python
- Pandas
- NumPy
- Scikit-learn
""")

with col2:
    st.markdown("""
- XGBoost
- Joblib
- Streamlit
- Matplotlib
""")

st.divider()

st.subheader("📂 Project Structure")

st.code(
"""
NIDAN
│
├── app
├── data
├── models
├── reports
├── src
├── requirements.txt
└── README.md
"""
)

st.divider()

st.success("Developed as an end-to-end Machine Learning project.")