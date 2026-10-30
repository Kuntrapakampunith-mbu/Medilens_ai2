import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json  # Added for JSON parsing of AI-extracted data
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="MediLens AI | Advanced Clinical Triage & RAG",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [CSS STYLING - Kept original] ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #0e1117 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #111622;
        padding: 10px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a2130;
        border-radius: 8px;
        color: #8b949e;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid #30363d;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4b4b, #ff6b6b) !important;
        color: white !important;
        border-color: #ff4b4b !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background-color: #1f242d;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #7a1c1c, #b32d2d);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(179, 45, 45, 0.5);
        border: 1px solid #ff6b6b;
    }
    .risk-low {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.5);
        border: 1px solid #4caf50;
    }
    .report-card {
        background: linear-gradient(145deg, #161b22, #1f242d);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #30363d;
        color: #e6edf3;
        font-size: 16px;
        line-height: 1.7;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .feature-badge {
        background-color: #1f6feb22;
        border: 1px solid #1f6feb;
        color: #58a6ff;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- [Initialize Session State for Dual-Mode Input] ---
feature_defaults = {
    'age': 54, 'sex': 1, 'trestbps': 130, 'chol': 246, 'cp': 0, 'fbs': 0,
    'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2
}

for key, default in feature_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --- [HELPER FUNCTIONS] ---
@st.cache_resource
def load_model():
    model_path = "model/heart_disease_model.pkl"
    features_path = "model/model_features.pkl"
    if os.path.exists(model_path) and os.path.exists(features_path):
        return joblib.load(model_path), joblib.load(features_path)
    return None, None

def load_rag_knowledge():
    if os.path.exists("knowledge_base.txt"):
        with open("knowledge_base.txt", "r") as f:
            return f.read()
    return "Standard cardiovascular prevention guidelines apply."

# Initialize resources
model, model_features = load_model()
knowledge_base_context = load_rag_knowledge()

# Header Section
col_h1, col_h2 = st.columns([3, 2])
with col_h1:
    st.title("🫀 MediLens AI")
    st.markdown("Advanced Clinical Triage & RAG Intelligence Platform")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(' <span class="feature-badge">⚡ Groq LPU Powered</span> &nbsp; <span class="feature-badge">🛡️ Guidelines RAG</span>', unsafe_allow_html=True)

st.markdown("---")

if model is None:
    st.error("⚠️ Model artifacts not found! Please run `python train_model.py` first.")
else:
    # --- [Hybrid Sidebar Inputs] ---
    st.sidebar.markdown("### 📋 Patient Input Method")
    input_mode = st.sidebar.radio(
        "Choose how to enter clinical data:",
        ["Manual Sliders (Default)", "AI Clinical Report Upload"]
    )
    st.sidebar.markdown("---")

    # Mode A: Automated Report Upload via Groq Extraction
    if input_mode == "AI Clinical Report Upload":
        st.sidebar.subheader("Automated AI Extraction")
        st.sidebar.info("Upload a patient clinical summary to automatically populate the parameters below.")
        
        uploaded_file = st.sidebar.file_uploader(
            "Drop Patient Report Here (.txt)", type=["txt"]
        )
        
        extraction_api_key = st.sidebar.text_input("Temporary Groq Key for Parsing:", type="password", key="parser_key")

        if uploaded_file is not None and st.sidebar.button("⚡ Extract Patient Vitals"):
            if not extraction_api_key:
                st.sidebar.warning("Please provide a Groq key for document parsing.")
            else:
                with st.sidebar.spinner("Groq AI is parsing the clinical report..."):
                    try:
                        report_text = uploaded_file.read().decode("utf-8")
                        
                        extraction_client = OpenAI(
                            api_key=extraction_api_key,
                            base_url="https://api.groq.com/openai/v1"
                        )
                        
                        prompt = f"""
                        You are a strict medical data parser. Read the patient medical report provided below and extract the 13 parameters required for a heart disease risk model as a single valid JSON object. Do not include any text outside the JSON.

                        Required parameters and data formats:
                        - age (integer, years 29-77)
                        - sex (integer, 1 for Male, 0 for Female)
                        - trestbps (integer, resting blood pressure 94-200 mm Hg)
                        - chol (integer, serum cholesterol 126-564 mg/dl)
                        - thalach (integer, maximum heart rate 71-202 bpm)
                        - oldpeak (float, ST depression 0.0-6.2)
                        - cp (integer, chest pain type 0-3)
                        - fbs (integer, fasting blood sugar > 120 mg/dl, 1 for True, 0 for False)
                        - restecg (integer, resting ECG results 0-2)
                        - exang (integer, exercise induced angina, 1 for Yes, 0 for No)
                        - slope (integer, slope of peak exercise ST segment 0-2)
                        - ca (integer, number of major vessels colored by fluoroscopy 0-3)
                        - thal (integer, thalassemia status 1-3)

                        Medical Report Text:
                        --------------------
                        {report_text}
                        --------------------

                        Return only the resulting valid JSON.
                        """
                        
                        response = extraction_client.chat.completions.create(
                            model="openai/gpt-oss-20b",  # Updated to active Groq model
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0
                        )
                        
                        extracted_json = response.choices[0].message.content.strip()
                        extracted_json = extracted_json.replace("```json", "").replace("```", "").strip()
                        extracted_data = json.loads(extracted_json)
                        
                        for key in feature_defaults.keys():
                            if key in extracted_data:
                                if key == 'oldpeak':
                                    st.session_state[key] = float(extracted_data[key])
                                else:
                                    st.session_state[key] = int(extracted_data[key])
                        
                        st.sidebar.success("✅ Extraction Complete! Check values below.")
                        
                    except Exception as e:
                        st.sidebar.error(f"Error parsing report. Ensure file contains clear vitals. Details: {e}")
        st.sidebar.markdown("---")

    # Mode B: Manual Entry (Sliders & Selectboxes connected to session state)
    st.sidebar.markdown("### 🔍 Parameters (Review & Edit)")
    st.sidebar.markdown("Adjust parameters below to refine diagnostic simulation.")

    with st.sidebar.expander("👤 Demographics & Vitals", expanded=True):
        age = st.slider("Age (Years)", 29, 77, value=st.session_state.age, key="age")
        
        sex_options = [1, 0]
        sex_index = 0 if st.session_state.sex == 1 else 1
        sex = st.selectbox("Sex", options=sex_options, format_func=lambda x: "Male" if x == 1 else "Female", index=sex_index, key="sex_select")
        st.session_state.sex = sex
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, value=st.session_state.trestbps, key="trestbps")
        chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, value=st.session_state.chol, key="chol")

    with st.sidebar.expander("🩺 Cardiac Diagnostics", expanded=False):
        cp_options = [0, 1, 2, 3]
        cp_func = lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]
        cp = st.selectbox("Chest Pain Type (0-3)", options=cp_options, format_func=cp_func, index=st.session_state.cp, key="cp_select")
        st.session_state.cp = cp

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False", index=(0 if st.session_state.fbs == 1 else 1), key="fbs_select")
        st.session_state.fbs = fbs
        
        restecg_options = [0, 1, 2]
        restecg_func = lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x]
        restecg = st.selectbox("Resting ECG Results", options=restecg_options, format_func=restecg_func, index=st.session_state.restecg, key="restecg_select")
        st.session_state.restecg = restecg
        
        thalach = st.slider("Max Heart Rate Achieved", 71, 202, value=st.session_state.thalach, key="thalach")

    with st.sidebar.expander("⚡ Exercise Stress Test", expanded=False):
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=(0 if st.session_state.exang == 1 else 1), key="exang_select")
        st.session_state.exang = exang
        
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, value=float(st.session_state.oldpeak), key="oldpeak")
        
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], index=st.session_state.slope, key="slope_select")
        st.session_state.slope = slope
        
        ca = st.selectbox("Major Vessels Colored (Fluoroscopy)", options=[0, 1, 2, 3], index=st.session_state.ca, key="ca_select")
        st.session_state.ca = ca
        
        thal_options = [1, 2, 3]
        thal_func = lambda x: ["", "Normal", "Fixed Defect", "Reversible Defect"][x]
        thal = st.selectbox("Thalassemia", options=thal_options, format_func=thal_func, index=st.session_state.thal - 1 if st.session_state.thal > 0 else 1, key="thal_select")
        st.session_state.thal = thal

    # --- [WORKFLOW - Assemble Data & Run Tabs] ---
    current_input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    input_df = pd.DataFrame([current_input_data])

    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Live Triage & Analysis", "🤖 RAG & AI Clinical Report", "📊 Structured Record", "ℹ️ Model Metrics"])

    with tab1:
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("### 🔍 Patient Snapshot")
            m1, m2, m3 = st.columns(3)
            m1.metric("Age / Sex", f"{age} yrs", "Male" if sex==1 else "Female")
            m2.metric("Blood Pressure", f"{trestbps} mmHg", "Resting")
            m3.metric("Cholesterol", f"{chol} mg/dl", "Serum")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Execute Risk Analysis", type="primary"):
                for col in model_features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                eval_df = input_df[model_features]
                
                prediction = model.predict(eval_df)[0]
                proba = model.predict_proba(eval_df)[0]
                
                st.session_state['pred_result'] = prediction
                st.session_state['pred_proba'] = proba

        with col2:
            st.markdown("### 🎯 Diagnostic Panel")
            if 'pred_result' in st.session_state:
                pred = st.session_state['pred_result']
                prob = st.session_state['pred_proba']
                
                if pred == 1:
                    st.markdown(f"""
                        <div class="risk-high">
                            <h2>⚠️ HIGH RISK DETECTED</h2>
                            <p style="font-size: 18px; margin: 0;">Probability: <b>{prob[1]*100:.1f}%</b></p>
                            <hr style="border-color: rgba(255,255,255,0.2);">
                            <p style="font-size: 14px; margin: 0;">Immediate clinical cardiology referral recommended.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="risk-low">
                            <h2>✅ LOW RISK PROFILE</h2>
                            <p style="font-size: 18px; margin: 0;">Confidence: <b>{prob[0]*100:.1f}%</b></p>
                            <hr style="border-color: rgba(255,255,255,0.2);">
                            <p style="font-size: 14px; margin: 0;">No immediate signs of heart disease detected.</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("👈 Fill out patient data in the sidebar (manually or via AI upload) and click **'Execute Risk Analysis'**.")

    with tab2:
        st.markdown("### 🤖 RAG-Powered AI Clinical Report")
        st.write("Synthesize an evidence-based medical summary by combining machine learning outputs with local cardiology guidelines via Groq.")
        
        main_groq_api_key = st.text_input("Enter main Groq API Key (gsk_...) for RAG Generation:", type="password", key="main_api_key")
        
        if st.button("⚡ Generate Evidence-Based Clinical Summary"):
            if not main_groq_api_key:
                st.warning("Please enter your Groq API key.")
            elif 'pred_result' not in st.session_state:
                st.warning("Please run the risk analysis in Tab 1 first!")
            else:
                with st.spinner("Retrieving knowledge base guidelines and querying Groq LLM..."):
                    try:
                        client = OpenAI(
                            api_key=main_groq_api_key,
                            base_url="https://api.groq.com/openai/v1"
                        )
                        
                        risk_str = "HIGH RISK" if st.session_state['pred_result'] == 1 else "LOW RISK"
                        conf = st.session_state['pred_proba'][st.session_state['pred_result']] * 100
                        
                        prompt = f"""
                        You are a professional clinical cardiology AI assistant.
                        
                        Retrieved Medical Guidelines (RAG Context):
                        {knowledge_base_context}
                        
                        Patient Parameters:
                        - Demographics: Age {st.session_state.age}, Sex {"Male" if st.session_state.sex==1 else "Female"}
                        - Vitals: Resting BP = {st.session_state.trestbps} mm Hg, Cholesterol = {st.session_state.chol} mg/dl
                        - Cardiac Findings: Max Heart Rate = {st.session_state.thalach}, ST Depression (Oldpeak) = {st.session_state.oldpeak}
                        - ML Risk Assessment: {risk_str} ({conf:.1f}% confidence)
                        
                        Instructions:
                        Draft a concise, professional clinical evaluation report based on these parameters.
                        1. Summarize the patient's critical health risk factors based on their vitals.
                        2. Cross-reference their parameters directly with the retrieved medical guidelines in the context.
                        3. Provide 3 specific, actionable recommendations for the clinician based on guidelines.
                        """
                        
                        response = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2
                        )
                        
                        summary = response.choices[0].message.content
                        st.markdown(f'<div class="report-card">{summary}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error querying Groq API: {e}")

    with tab3:
        st.markdown("### 📋 Structured Patient Data Record")
        st.dataframe(input_df, width='stretch')

    with tab4:
        st.markdown("### 🧠 Random Forest Model Architecture")
        st.write("The underlying classifier is trained on the UCI Cleveland Heart Disease dataset achieving high optimized cross-validation accuracy combined with an external RAG guidelines layer.")
        col_a, col_b = st.columns(2)
        col_a.metric("Model Type", "Random Forest Classifier")
        col_b.metric("Reported Accuracy", "88.3%")