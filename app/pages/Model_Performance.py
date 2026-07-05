import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Performance")

report_path = Path(__file__).resolve().parents[2] / "reports" / "model_comparison.csv"

results = pd.read_csv(report_path)
results = results.sort_values(by="ROC_AUC", ascending=False)

best = results.iloc[0]

st.subheader("🏆 Best Model")

col1, col2, col3 = st.columns(3)

col1.metric("Model", best["Model"])
col2.metric("Accuracy", f"{best['Accuracy']:.3f}")
col3.metric("ROC-AUC", f"{best['ROC_AUC']:.3f}")

st.divider()

st.subheader("📋 Model Comparison")

st.write(results)

st.divider()

st.subheader("📈 Accuracy Comparison")
st.bar_chart(results.set_index("Model")[["Accuracy"]])

st.subheader("🎯 F1 Score Comparison")
st.bar_chart(results.set_index("Model")[["F1 Score"]])

st.subheader("📊 ROC-AUC Comparison")
st.bar_chart(results.set_index("Model")[["ROC_AUC"]])