"""
Air Quality Category Predictor — Streamlit App
================================================
Run locally:
    streamlit run app.py

Requires aqi_model.pkl, label_encoder.pkl, feature_cols.pkl
(produced by train_model.py) to be in the same folder.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌫️", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("aqi_model.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, le, feature_cols

model, le, feature_cols = load_artifacts()

st.title("🌫️ Air Quality Category Predictor")
st.write(
    "Enter sensor readings below to predict the air quality category "
    "(Good / Moderate / Poor / Severe), based on a model trained on the "
    "UCI Air Quality dataset."
)

st.markdown("---")

# Friendly labels + reasonable default values for each raw feature
friendly_names = {
    "CO(GT)": ("CO concentration (mg/m³)", 1.5),
    "PT08.S1(CO)": ("CO sensor response (PT08.S1)", 1100.0),
    "NMHC(GT)": ("Non-methane hydrocarbons (µg/m³)", 150.0),
    "C6H6(GT)": ("Benzene (µg/m³)", 8.0),
    "PT08.S2(NMHC)": ("NMHC sensor response (PT08.S2)", 900.0),
    "NOx(GT)": ("NOx concentration (ppb)", 150.0),
    "PT08.S3(NOx)": ("NOx sensor response (PT08.S3)", 900.0),
    "NO2(GT)": ("NO2 concentration (µg/m³)", 70.0),
    "PT08.S4(NO2)": ("NO2 sensor response (PT08.S4)", 1400.0),
    "PT08.S5(O3)": ("O3 sensor response (PT08.S5)", 900.0),
    "T": ("Temperature (°C)", 20.0),
    "RH": ("Relative Humidity (%)", 50.0),
    "AH": ("Absolute Humidity", 1.0),
}

inputs = {}
cols = st.columns(2)
for i, col_name in enumerate(feature_cols):
    label, default = friendly_names.get(col_name, (col_name, 0.0))
    with cols[i % 2]:
        inputs[col_name] = st.number_input(label, value=float(default))

st.markdown("---")

if st.button("Predict Air Quality Category", type="primary"):
    X_new = pd.DataFrame([inputs])[feature_cols]
    pred_encoded = model.predict(X_new)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(X_new)[0]

    color_map = {"Good": "🟢", "Moderate": "🟡", "Poor": "🟠", "Severe": "🔴"}
    st.subheader(f"{color_map.get(pred_label, '')} Predicted category: **{pred_label}**")

    proba_df = pd.DataFrame({"Category": le.classes_, "Probability": proba}).sort_values(
        "Probability", ascending=False
    )
    st.bar_chart(proba_df.set_index("Category"))

st.markdown("---")
st.caption(
    "Note: This is a simplified proxy AQI category (based on CO and NO2 "
    "breakpoints), not an official EPA/CPCB AQI, since the underlying "
    "UCI dataset does not include PM2.5/PM10 readings."
)
