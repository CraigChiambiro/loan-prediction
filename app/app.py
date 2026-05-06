# Importing all the libraries
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd
from reportlab.pdfgen import canvas
import io

# =========================
# LOAD MODEL + SCALER
# =========================

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# PAGE CONFIG (DARK STYLE)
# =========================

st.set_page_config(
    page_title="Loan AI System",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
    <style>
        body {background-color: #0e1117;}
        .stApp {background-color: #0e1117;}
    </style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("🏦 AI Loan Approval System")
st.write("Professional ML Decision Engine")

st.divider()

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
married = st.sidebar.selectbox("Married", ["No", "Yes"])
dependents = st.sidebar.slider("Dependents", 0, 3)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

income = st.sidebar.number_input("Income", 0)
co_income = st.sidebar.number_input("Coapplicant Income", 0)
loan_amount = st.sidebar.number_input("Loan Amount", 0)
loan_term = st.sidebar.number_input("Loan Term", 0)

credit = st.sidebar.selectbox("Credit History", ["Bad", "Good"])
area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# =========================
# ENCODING
# =========================

maps = {
    "gender": {"Female": 0, "Male": 1},
    "married": {"No": 0, "Yes": 1},
    "education": {"Graduate": 0, "Not Graduate": 1},
    "self": {"No": 0, "Yes": 1},
    "credit": {"Bad": 0, "Good": 1},
    "area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
}

# =========================
# FEATURE IMPORTANCE (APPROX)
# =========================

feature_names = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "Income","CoIncome","LoanAmount","LoanTerm","Credit","Area"
]

# =========================
# PREDICTION FUNCTION
# =========================

def predict_all():
    input_data = np.array([[
        maps["gender"][gender],
        maps["married"][married],
        dependents,
        maps["education"][education],
        maps["self"][self_employed],
        income,
        co_income,
        loan_amount,
        loan_term,
        maps["credit"][credit],
        maps["area"][area]
    ]])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][prediction]

    return prediction, prob, input_data[0]

# =========================
# PDF REPORT FUNCTION
# =========================

def create_pdf(result, confidence):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)

    p.drawString(100, 800, "Loan Prediction Report")
    p.drawString(100, 780, f"Result: {'Approved' if result==1 else 'Rejected'}")
    p.drawString(100, 760, f"Confidence: {confidence:.2f}%")

    p.save()
    buffer.seek(0)
    return buffer

# =========================
# BUTTON
# =========================

if st.button("🚀 Predict Loan Status", use_container_width=True):

    result, prob, raw_input = predict_all()
    confidence = prob * 100

    st.divider()

    # =========================
    # RESULT
    # =========================

    if result == 1:
        st.success("🎉 Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric("Confidence", f"{confidence:.2f}%")

    # =========================
    # GAUGE
    # =========================

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={"text": "Approval Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, 40], "color": "red"},
                {"range": [40, 70], "color": "orange"},
                {"range": [70, 100], "color": "green"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # EXPLANATION (WHY RESULT)
    # =========================

    st.subheader("🧠 Decision Explanation")

    if result == 1:
        st.write("✔ Strong credit profile and income levels support approval")
    else:
        st.write("⚠ Low credit history or income may have influenced rejection")

    # =========================
    # FEATURE IMPORTANCE (SIMPLE VISUAL)
    # =========================

    st.subheader("📊 Input Overview")

    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": raw_input
    })

    st.bar_chart(df.set_index("Feature"))

    # =========================
    # PDF DOWNLOAD
    # =========================

    pdf = create_pdf(result, confidence)

    st.download_button(
        label="📄 Download Report",
        data=pdf,
        file_name="loan_report.pdf",
        mime="application/pdf"
    )

# =========================
# FOOTER
# =========================

st.divider()
st.caption("Built with ML + Streamlit | Portfolio Project")