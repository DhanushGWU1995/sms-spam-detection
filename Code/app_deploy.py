import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import load_distilbert, load_gru, load_logreg, score_message

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
        margin-top: 12px;
    }
    .flash-fail {
        background-color: #3c001a;
        padding: 16px;
        border-radius: 10px;
        margin-top: 12px;
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
def load_models():
    return load_logreg(), load_gru(), load_distilbert()


st.markdown("<h1 style='text-align: center;'>SMS Spam Filter</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color:#caf0f8;'>"
    "Paste a text. We show logistic regression, the GRU, and DistilBERT on the same message."
    "</p>",
    unsafe_allow_html=True,
)

logreg, gru, distilbert = load_models()
if logreg is None and gru is None and distilbert is None:
    st.error(
        "No saved models are in Code/artifacts yet. "
        "Open Code/main.ipynb and run it once, then restart this app."
    )
    st.stop()

st.markdown("#### Text message")
message = st.text_area("Message", height=220, label_visibility="collapsed")

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
        outcomes = score_message(message, logreg, gru, distilbert)
        cards = st.columns(3)
        for column, outcome in zip(cards, outcomes):
            with column:
                st.markdown(f"#### {outcome['name']}")
                st.caption(outcome["role"])
                if not outcome["ready"]:
                    st.info(outcome["detail"])
                    continue
                flash = "flash-fail" if outcome["spam"] else "flash-success"
                label = "Spam detected" if outcome["spam"] else "Message looks safe"
                st.markdown(
                    f'<div class="{flash}"><h3 style="text-align:center;">{label}</h3></div>',
                    unsafe_allow_html=True,
                )
                st.write(f"Spam probability: {outcome['probability']:.1%}")
                st.write(f"Threshold: {outcome['threshold']:.2f}")
        ready = [item for item in outcomes if item["ready"]]
        if len(ready) > 1:
            labels = {item["spam"] for item in ready}
            if len(labels) > 1:
                st.warning("The models disagree on this message.")
            else:
                decision = "spam" if ready[0]["spam"] else "ham"
                st.success(f"All trained models agree: {decision}.")
