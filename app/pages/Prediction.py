import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Patient Prediction", page_icon="🏥")

st.title("🏥 Patient Mortality Prediction")

model_path = Path(__file__).resolve().parents[2] / "models" / "best_model.pkl"

model = joblib.load(model_path)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.write("✅ File uploaded successfully")

    st.write("Shape:", data.shape)

    st.dataframe(data.head())

    if st.button("Predict"):

        st.write("Predict button clicked")

        predictions = model.predict(data)

        st.success("Prediction completed")

        st.write(predictions[:10])