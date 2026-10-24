import json
import os
import sys
import time

import joblib
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "logreg_model.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.json")

st.set_page_config(
    page_title="SMS Spam Filter",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .stTextArea textarea {
        background-color: #0b1d3a !important;
        color: #8ecae6 !important;
        font-family: monospace;
        font-size: 15px;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b4d8;
        color: black;
    }
    .flash-success {
        background-color: #003c1d;
        padding: 16px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .flash-fail {
        background-color: #3c001a;
        padding: 16px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .sample-card {
        background-color: #031b3f;
        border-left: 4px solid #00b4d8;
        padding: 12px 20px;
        margin-bottom: 10px;
        border-radius: 8px;
        color: #dcdcdc;
        font-size: 14px;
    }
    h1, h3, h4 {
        color: #90e0ef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    threshold = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, encoding="utf-8") as handle:
            threshold = float(json.load(handle).get("logreg", 0.5))
    return model, vectorizer, threshold


st.markdown("<h1 style='text-align: center;'>SMS Spam Filter</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color:#caf0f8;'>"
    "Paste a text message to classify it as spam or ham."
    "</p>",
    unsafe_allow_html=True,
)

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.error(
        "Model files are not in Code/artifacts yet. "
        "Open Code/main.ipynb and run it once to train the logistic regression baseline."
    )
    st.stop()

model, vectorizer, threshold = load_model_and_vectorizer()

st.markdown("#### Text message")
message = st.text_area(" ", height=220)

st.markdown("### Example messages")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Ham examples")
    st.markdown(
        '<div class="sample-card">Ok lar... Joking wif u oni...</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sample-card">Nah I don\'t think he goes to usf, he lives around here though</div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("#### Spam examples")
    st.markdown(
        '<div class="sample-card">Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. '
        "Text FA to 87121 to receive entry question(std txt rate)T&amp;C's apply 08452810075over18's</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sample-card">FreeMsg Hey there darling it\'s been 3 week\'s now and no word back! '
        "I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv</div>",
        unsafe_allow_html=True,
    )

st.markdown("### Classify message")
if st.button("Predict"):
    if not message.strip():
        st.warning("Paste an SMS first.")
    else:
        features = vectorizer.transform([preprocess_text(message)])
        spam_probability = float(model.predict_proba(features)[0, 1])
        is_spam = spam_probability >= threshold
        flash_class = "flash-fail" if is_spam else "flash-success"
        label = "Spam detected" if is_spam else "Message looks safe"
        st.markdown(
            f'<div class="{flash_class}"><h3 style="text-align:center;">{label}</h3></div>',
            unsafe_allow_html=True,
        )
        st.write(f"Spam probability: {spam_probability:.1%} (threshold {threshold:.2f})")
        time.sleep(1)
