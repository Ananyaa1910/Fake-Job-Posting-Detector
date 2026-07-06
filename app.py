import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    with open("fake_job_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts["model"]
tfidf = artifacts["tfidf"]

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()

def extract_meta(title, description):
    return np.array([[
        0,  # has_company_logo — unknown from text
        0,  # telecommuting
        1,  # missing_salary — we don't have it
        0,  # missing_company
        0,  # missing_requirements
        0,  # missing_benefits
        len(description),
        len(title),
        0,
        description.count("!"),
        description.count("?"),
        sum(1 for c in description if c.isupper()) / max(len(description), 1),
        int(bool(re.search(r"earn|income|\$|paid|weekly|daily|cash", title.lower()))),
        0,  # is_parttime
        0,  # is_contract
    ]])

def predict(title, description):
    combined = clean_text(title) + " " + clean_text(description)
    tfidf_vec = tfidf.transform([combined])
    meta_vec = csr_matrix(extract_meta(title, description))
    X = hstack([tfidf_vec, meta_vec])
    prob = model.predict_proba(X)[0][1]
    label = "🚨 LIKELY FAKE" if prob > 0.5 else "✅ LIKELY REAL"
    return label, prob

def highlight_suspicious(text):
    suspicious = [
        r"work from home", r"earn (\$[\d,]+|money)",
        r"no experience", r"immediate(ly)?", r"apply now",
        r"limited positions", r"guaranteed", r"wire transfer",
        r"weekly pay", r"\$(\d{3,})"
    ]
    highlighted = text
    for pattern in suspicious:
        highlighted = re.sub(
            pattern,
            lambda m: f"**:red[{m.group()}]**",
            highlighted, flags=re.IGNORECASE
        )
    return highlighted

# ── UI ─────────────────────────────────────────────────────────
st.title("🔍 Fake Job Posting Detector")
st.caption("Trained on EMSCAD dataset · 17,880 job postings · XGBoost + TF-IDF")
st.divider()

title = st.text_input("Job Title", placeholder="e.g. Data Entry Specialist – Work From Home")
description = st.text_area(
    "Job Description",
    placeholder="Paste the full job description here...",
    height=250
)

if st.button("Analyze Job Posting", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing..."):
            label, prob = predict(title, description)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Verdict", label)
        with col2:
            st.metric("Fraud Probability", f"{prob*100:.1f}%")

        st.progress(float(prob), text=f"Confidence: {prob*100:.1f}% fraudulent")

        if prob > 0.5:
            st.error("⚠️ This posting shows signs of fraud. Do not share personal or financial information.")
        else:
            st.success("This posting appears legitimate, but always research the company independently.")

        # Highlight suspicious phrases
        st.subheader("Suspicious Phrase Detection")
        highlighted = highlight_suspicious(description)
        st.markdown(highlighted)

st.divider()
st.caption("Built with XGBoost · TF-IDF · SHAP · Streamlit")