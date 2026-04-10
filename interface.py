import streamlit as st
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from io import BytesIO
from reportlab.pdfgen import canvas
import os

# ================= SAFE PATH LOADING =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "liver_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.set_page_config(page_title="Liver AI Pro", layout="wide")

# ================= SESSION STATE FIX =================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Overview"

# ================= STYLE =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;500;700&display=swap');

html, body {
    font-family: 'Cairo', sans-serif;
    background: radial-gradient(circle at top, #0b1220, #05070f);
    color: white;
}

/* HEADER */
.header {
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    margin-top: 10px;
    background: linear-gradient(90deg,#00c6ff,#0072ff,#00c6ff);
    background-size: 200% 200%;
    animation: move 4s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

@keyframes move {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.sub {
    text-align:center;
    color:#94a3b8;
    margin-bottom:20px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(0,198,255,0.3);
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    padding: 12px;
    border-radius: 12px;
    font-size: 18px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='header'>🧠 AI Liver Diagnostic Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Advanced Medical Decision Support System</div>", unsafe_allow_html=True)

st.write("---")

# ================= SIDEBAR =================
pages = ["🏠 Overview", "👨‍💻 Team", "📊 System Status", "📌 Project Info"]

menu = st.sidebar.selectbox("Navigation", pages, index=pages.index(st.session_state.page))

st.session_state.page = menu

st.sidebar.write("---")

if menu == "🏠 Overview":
    st.sidebar.success("System Running 🟢")

elif menu == "👨‍💻 Team":
    st.sidebar.info("AI Engineering Team Active")

elif menu == "📊 System Status":
    st.sidebar.metric("Model", "Active")
    st.sidebar.metric("Latency", "Low")

elif menu == "📌 Project Info":
    st.sidebar.warning("Educational Use Only")

# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin", 0.1, 50.0, 1.0)
    db = st.number_input("Direct Bilirubin", 0.1, 20.0, 0.5)
    ap = st.number_input("Alkaline Phosphotase", 10, 2000, 200)

with col2:
    sgpt = st.number_input("ALT (SGPT)", 10, 2000, 40)
    sgot = st.number_input("AST (SGOT)", 10, 2000, 40)
    tp = st.number_input("Total Proteins", 1.0, 10.0, 7.0)
    alb = st.number_input("Albumin", 1.0, 10.0, 3.5)
    ag_ratio = st.number_input("A/G Ratio", 0.1, 3.0, 1.0)

st.write("---")

run = st.button("🚀 Run AI Analysis")

# ================= PDF =================
def generate_pdf(status, risk, age):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)

    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 800, "Liver AI Diagnostic Report")

    p.setFont("Helvetica", 12)
    p.drawString(100, 760, f"Status: {status}")
    p.drawString(100, 740, f"Risk: {risk*100:.2f}%")
    p.drawString(100, 720, f"Age: {age}")

    p.save()
    buffer.seek(0)
    return buffer

# ================= MODEL =================
if run:

    with st.spinner("🧠 AI analyzing patient data..."):
        time.sleep(2)

    gender_num = 1 if gender == "Male" else 0

    features = np.array([[age, gender_num, tb, db, ap, sgpt, sgot, tp, alb, ag_ratio]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features_scaled)[0][prediction]
    else:
        prob = 0.7

    risk = float(prob)

    status = "High Risk" if prediction == 1 else "Low Risk"

    st.write("---")

    if prediction == 1:
        st.error("⚠️ HIGH RISK DETECTED")
    else:
        st.success("✅ LOW RISK - NORMAL")

    # ================= GAUGE =================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00c6ff"},
            'steps': [
                {'range': [0, 40], 'color': "#22c55e"},
                {'range': [40, 70], 'color': "#facc15"},
                {'range': [70, 100], 'color': "#ef4444"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.progress(risk)

    # ================= PDF DOWNLOAD =================
    pdf = generate_pdf(status, risk, age)

    st.download_button(
        label="📄 Download Medical Report (PDF)",
        data=pdf,
        file_name="liver_report.pdf",
        mime="application/pdf"
    )