import streamlit as st

st.set_page_config(
    page_title="NIDAN",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 NIDAN")
st.subheader("In-Hospital Mortality Prediction System")

st.write(
    """
NIDAN is an end-to-end machine learning application that predicts
the risk of in-hospital mortality using patient clinical data.

The project compares multiple machine learning models, selects the
best-performing model based on ROC-AUC score, and provides predictions
through an interactive Streamlit web application.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Models Compared", "6")

with col2:
    st.metric("Best Model", "Extra Trees")

with col3:
    st.metric("Deployment", "Streamlit")

st.divider()

st.header("📌 Project Workflow")

st.markdown("""
- Data Collection
- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Comparison
- Model Selection
- Prediction
- Web Deployment
""")

st.divider()

st.header("🛠 Technologies")

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

st.info(
    "Use the sidebar to navigate to Prediction, Model Performance, and About pages."
)