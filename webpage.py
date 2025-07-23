import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
from predict import predict
from PIL import Image
import base64
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="My Heart Health", layout="centered")

# --- Theme Settings (Dark default) ---
theme = st.sidebar.radio("🌓 Choose Theme", ["Dark", "Light"], index=0)
text_color = "#FFFFFF" if theme == "Dark" else "#000000"
bg_color = "#1e1e1e" if theme == "Dark" else "#ffffff"
input_bg = "#333" if theme == "Dark" else "#e0e0e0"

# --- Custom Styling ---
st.markdown(f"""
    <style>
        body {{ color: {text_color}; }}
        .main {{
            background-color: {bg_color};
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }}
        input, select {{
            background-color: {input_bg};
            color: {text_color};
            border: 2px solid red;
            border-radius: 8px;
            padding: 8px;
        }}
        label, .stSelectbox label, .stNumberInput label, .stRadio label, .css-1y4p8pa {{
            color: {text_color} !important;
            font-weight: bold;
        }}
        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #d73838;
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        }}
        .result-box {{
            background-color: #fce4ec;
            border-left: 5px solid #e91e63;
            padding: 20px;
            margin-top: 20px;
            border-radius: 12px;
            font-size: 18px;
            color: black;
        }}
        .pdf-button {{
            background-color: #4CAF50;
            color: white !important;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 10px;
            display: inline-block;
            margin-top: 20px;
            font-weight: bold;
        }}
        .pdf-button:hover {{
            background-color: #45a049;
        }}
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}
        .title-text {{
            font-size: 40px;
            font-family: 'Brush Script MT', cursive;
            font-weight: bold;
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# --- Logo & Title ---
try:
    logo = Image.open("logo.ico")
    logo_html = f"""
    <div class='title-container'>
        <img src="data:image/png;base64,{base64.b64encode(open('logo.ico', 'rb').read()).decode()}" width="60">
        <div class='title-text'>My Heart Health</div>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)
except:
    st.markdown(f"<h1 style='text-align: center; color:{text_color}'>My Heart Health</h1>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("heart_form"):
    st.markdown(f"<h3 style='color:{text_color}'>📝 Enter Your Health Information</h3>", unsafe_allow_html=True)
    age = st.number_input("1. Age (Years)", min_value=1, max_value=100, step=1)
    sex = st.selectbox("2. Sex", ["Male", "Female"])
    race = st.selectbox("3. Race", ["Indian", "White", "Black", "Asian", "Hispanic", "Other"])
    weight = st.number_input("4. Weight (kg)", min_value=30.0, max_value=200.0, step=0.5)
    height = st.number_input("5. Height (cm)", min_value=0.0, max_value=250.0, step=0.5)
    smoking = st.selectbox("6. Smoking", ["No", "Yes"])
    alcohol = st.selectbox("7. Alcohol Drinking", ["No", "Yes"])
    diabetic = st.selectbox("8. Diabetic", ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])
    diff_walking = st.selectbox("9. Difficulty Walking", ["No", "Yes"])
    physical_activity = st.selectbox("10. Physical Activity", ["No", "Yes"])
    gen_health = st.selectbox("11. General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
    sleep_time = st.number_input("12. Sleep Time (hours)", min_value=0, max_value=24, value=8)
    stroke = st.selectbox("13. Stroke", ["No", "Yes"])
    asthma = st.selectbox("14. Asthma", ["No", "Yes"])
    kidney_disease = st.selectbox("15. Kidney Disease", ["No", "Yes"])
    skin_cancer = st.selectbox("16. Skin Cancer", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict My Heart Risk")

if submitted:
    try:
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)

        input_data = {
            "BMI": bmi,
            "Smoking": smoking,
            "AlcoholDrinking": alcohol,
            "Stroke": stroke,
            "DiffWalking": diff_walking,
            "Sex": sex,
            "AgeCategory": str(age),
            "Race": race,
            "Diabetic": diabetic,
            "PhysicalActivity": physical_activity,
            "GenHealth": gen_health,
            "SleepTime": sleep_time,
            "Asthma": asthma,
            "KidneyDisease": kidney_disease,
            "SkinCancer": skin_cancer
        }

        results = predict(input_data)
        ensemble_result = results["Ensemble"]
        label = ensemble_result["label"]
        prob = ensemble_result["probability"]

        disease_probs = {
            "Coronary Artery Disease": round(prob * 0.65, 2),
            "Arrhythmia": round(prob * 0.2, 2),
            "Heart Failure": round(prob * 0.1, 2),
            "Valve Disease": round(prob * 0.05, 2)
        }

        consult = "Please consult a doctor." if label == "Heart Disease" else "No immediate concern."

        st.markdown(f"<h3 style='color:{text_color}'>📊 Prediction Result</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='result-box'>
                <b>{'High Risk' if label == 'Heart Disease' else 'Low Risk'} ({prob}%)</b><br>
                {consult}<br><br>
                <b>Heart Disease Risk:</b> {prob}%<br><br>
                <b>Other Potential Conditions:</b><br>
                {'<br>'.join([f"• {k}: {v}%" for k, v in disease_probs.items()])}
            </div>
        """, unsafe_allow_html=True)

        # --- PDF Report ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Heart Disease Risk Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt="User Inputs:", ln=True)
        for key, val in input_data.items():
            pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if label == 'Heart Disease' else 'Low Risk'} ({prob}%)", ln=True)
        pdf.cell(200, 10, txt=consult, ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt="Other Potential Conditions:", ln=True)
        for disease, p in disease_probs.items():
            pdf.cell(200, 10, txt=f"- {disease}: {p}%", ln=True)

        pdf_output = "heart_risk_report.pdf"
        pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            pdf_data = f.read()
            b64_pdf = base64.b64encode(pdf_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="heart_risk_report.pdf" class="pdf-button">📄 Download Full Report (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
