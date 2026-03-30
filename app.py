import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib



st.set_page_config(page_title="Anemia Diagnosis System", layout="centered")
st.title("🩸 Anemia Diagnosis Assistant")
st.write("Enter the CBC (Complete Blood Count) values below to get a diagnosis prediction.")

@st.cache_resource
def load_anemia_artifacts():
    model = load_model("models/anemia_model.keras")
    scaler = joblib.load("models/anemia_scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler , label_encoder

model, scaler , label_encoder = load_anemia_artifacts()

st.subheader("Patient Blood Values")
col1 , col2 = st.columns(2)

with col1:
    wbc = st.number_input("WBC (White Blood Cell Count)", value=9.4)
    lymp = st.number_input("LYMp (Lymphocyte Percentage %)", value=34.0)
    neutp = st.number_input("NEUTp (Neutrophil Percentage %)", value=51.0)
    lymn = st.number_input("LYMn (Lymphocyte Count #)", value=4.1)
    neutn = st.number_input("NEUTn (Neutrophil Count #)", value=5.0)
    rbc = st.number_input("RBC (Red Blood Cell Count)", value=2.62)
    hgb = st.number_input("HGB (Hemoglobin)", value=7.1)

with col2:
    hct = st.number_input("HCT (Hematocrit)", value=27.3)
    mcv = st.number_input("MCV (Mean Corpuscular Volume)", value=89.2)
    mch = st.number_input("MCH (Mean Corpuscular Hemoglobin)", value=25.1)
    mchc = st.number_input("MCHC (Mean Corpuscular Hemoglobin Concentration)", value=31.1)
    plt = st.number_input("PLT (Platelet Count)", value=187.1)
    pdw = st.number_input("PDW (Platelet Distribution Width)", value=12.7)
    pct = st.number_input("PCT (Plateletcrit)", value=0.16)

if st.button("Predict Diagnosis",type="primary"):
    column_names = [
        "WBC","LYMp","NEUTp","LYMn","NEUTn","RBC","HGB","HCT","MCV","MCH","MCHC","PLT","PDW","PCT"
    ]
    input_df = pd.DataFrame([[wbc, lymp, neutp, lymn, neutn, rbc, hgb,
                              hct, mcv, mch, mchc, plt, pdw, pct]],
                            columns=column_names)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_class_index = np.argmax(prediction)
    diagnosis_name = label_encoder.inverse_transform([predicted_class_index])[0]

    st.markdown("---")
    st.subheader("Result:")
    st.info(f"Final Diagnosis: **{diagnosis_name}**")
    st.warning("Please consult a medical professional for a definitive diagnosis.")