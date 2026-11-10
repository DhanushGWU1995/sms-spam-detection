import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (
    ARTIFACT_DIR,
    load_distilbert,
    load_gru,
    load_logreg,
    score_distilbert,
    score_gru,
    score_logreg,
)
from llm_cascade import final_decision, label_message, load_qwen, saved_weak_margin

APPROACHES = ("Traditional models", "LLM only", "Hybrid")
DATASET = {
    "messages": 67709,
    "ham": 41390,
    "spam": 26319,
    "train": 54167,
    "validation": 6771,
    "test": 6771,
}
OPTIONS = (
    "Hybrid",
    "Traditional models",
    "Logistic regression",
    "GRU",
    "DistilBERT",
    "LLM only",
)
OPTION_NOTES = {
    "Hybrid": "The three trained models vote. Qwen is called only when DistilBERT's confidence is weak.",
    "Traditional models": "Majority vote of logistic regression, the GRU, and DistilBERT. Qwen is not called.",
    "Logistic regression": "TF-IDF features and a balanced logistic regression. Qwen is not called.",
    "GRU": "A PyTorch GRU trained on the cleaned SMS text. Qwen is not called.",
    "DistilBERT": "Fine-tuned DistilBERT on the raw SMS text. Qwen is not called.",
    "LLM only": "Qwen2.5-1.5B labels the message by itself. The trained models are not used.",
}

st.set_page_config(page_title="SMS Spam Detection", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def load_models():
    return load_logreg(), load_gru(), load_distilbert()


@st.cache_resource
def load_qwen_model():
    return load_qwen()


@st.cache_data
def load_reports():
    path = ARTIFACT_DIR / "approach_reports.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_word(probability, threshold, margin):
    if abs(probability - threshold) <= margin:
        return "weak"
    return "strong"


def missing_model(name, role):
    return {
        "name": name,
        "role": role,
        "ready": False,
        "detail": "This model is not saved yet. Run the notebook, then restart the app.",
    }


def describe_outcome(outcome, margin):
    if not outcome["ready"]:
        return f"{outcome['name']}: unavailable. {outcome.get('detail', '')}"
    label = "spam" if outcome["spam"] else "ham"
    strength = confidence_word(outcome["probability"], outcome["threshold"], margin)
    return (
        f"{outcome['name']} predicts {label}. "
        f"Spam probability {outcome['probability']:.1%}. "
        f"Confidence is {strength}."
    )


def score_named(name, bundle, scorer, role, message):
    if bundle is None:
        return missing_model(name, role)
    return scorer(bundle, message)


def ask_qwen(message, status):
    status.write("Qwen is triggering.")
    try:
        qwen = label_message(message, load_qwen_model())
    except Exception as exc:
        status.write(f"Qwen did not load ({exc}).")
        return None, str(exc)
    if qwen.get("ready"):
        answered = "spam" if qwen["spam"] else "ham"
        status.write(f"Qwen answered {answered}.")
    else:
        status.write("Qwen did not answer spam or ham.")
    return qwen, ""


def predict(option, message, logreg, gru, distilbert, status):
    margin = saved_weak_margin()
    outcomes = []
    qwen = None
    qwen_error = ""
    weak = False

    if option == "LLM only":
        status.write("This option uses Qwen only.")
        qwen, qwen_error = ask_qwen(message, status)
        spam = qwen.get("spam") if qwen and qwen.get("ready") else None
        return {
            "option": option,
            "outcomes": outcomes,
            "qwen": qwen,
            "qwen_error": qwen_error,
            "margin": margin,
            "weak": False,
            "decision": {"spam": spam, "called_qwen": True},
        }

    wanted = {
        "Logistic regression": [("Logistic Regression", logreg, score_logreg, "TF-IDF baseline")],
        "GRU": [("GRU", gru, score_gru, "Recurrent network")],
        "DistilBERT": [("DistilBERT", distilbert, score_distilbert, "Transformer")],
    }
    if option in wanted:
        steps = wanted[option]
    else:
        steps = [
            ("Logistic Regression", logreg, score_logreg, "TF-IDF baseline"),
            ("GRU", gru, score_gru, "Recurrent network"),
            ("DistilBERT", distilbert, score_distilbert, "Transformer"),
        ]

    for name, bundle, scorer, role in steps:
        status.write(f"Scoring {name}.")
        outcome = score_named(name, bundle, scorer, role, message)
        outcomes.append(outcome)
        status.write(describe_outcome(outcome, margin))

    bert = next((item for item in outcomes if item["name"] == "DistilBERT" and item["ready"]), None)
    weak = bert is not None and confidence_word(bert["probability"], bert["threshold"], margin) == "weak"
    if option == "Hybrid" and weak:
        status.write("DistilBERT confidence is weak.")
        qwen, qwen_error = ask_qwen(message, status)
        if qwen and qwen.get("ready"):
            decision = final_decision(outcomes, weak=True, qwen_spam=qwen.get("spam"))
        else:
            status.write("The traditional vote stands.")
            decision = final_decision(outcomes, weak=False)
    else:
        if option == "Hybrid":
            status.write("DistilBERT confidence is strong. Qwen is not triggered.")
        else:
            status.write("Qwen is not used for this option.")
        decision = final_decision(outcomes, weak=False)

    return {
        "option": option,
        "outcomes": outcomes,
        "qwen": qwen,
        "qwen_error": qwen_error,
        "margin": margin,
        "weak": weak and option == "Hybrid",
        "decision": decision,
    }


def render_response(result):
    decision = result["decision"]
    if decision.get("spam") is None:
        st.warning("This option could not produce a label.")
        if result["qwen_error"]:
            st.info(result["qwen_error"])
        return
    label = "spam" if decision["spam"] else "ham"
    if result["option"] == "LLM only":
        how = "Response from Qwen only."
    elif result["option"] == "Hybrid" and result["qwen"] and result["qwen"].get("ready"):
        how = "Response from Qwen, because DistilBERT's confidence was weak."
    elif result["option"] == "Hybrid":
        how = "Response from the majority vote. DistilBERT's confidence was strong."
    elif result["option"] == "Traditional models":
        how = "Response from the majority vote of the three trained models."
    else:
        how = f"Response from {result['option']} only."
    st.markdown(f"### Response: {label}")
    st.write(how)

    rows = []
    for outcome in result["outcomes"]:
        if not outcome["ready"]:
            rows.append({"Model": outcome["name"], "Label": "unavailable", "Spam probability": "—", "Confidence": "—"})
            continue
        rows.append(
            {
                "Model": outcome["name"],
                "Label": "spam" if outcome["spam"] else "ham",
                "Spam probability": f"{outcome['probability']:.1%}",
                "Confidence": confidence_word(outcome["probability"], outcome["threshold"], result["margin"]),
            }
        )
    if result["qwen"] is not None:
        qwen = result["qwen"]
        rows.append(
            {
                "Model": "Qwen",
                "Label": "spam" if qwen.get("spam") else "ham" if qwen.get("ready") else "unparsed",
                "Spam probability": "—",
                "Confidence": "triggered" if qwen.get("ready") else "unparsed",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def metric_frame(split_payload):
    rows = []
    for name in APPROACHES:
        row = split_payload["approaches"][name]
        rows.append(
            {
                "Approach": name,
                "F1": row["f1"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "Accuracy": row["accuracy"],
            }
        )
    return pd.DataFrame(rows).set_index("Approach")


def confusion_figure(matrix, title):
    import plotly.graph_objects as go

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted ham", "Predicted spam"],
            y=["Actual ham", "Actual spam"],
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale="Blues",
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>count=%{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=320,
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        yaxis={"autorange": "reversed"},
    )
    return figure


def render_dataset():
    st.subheader("Dataset")
    counts = st.columns(6)
    labels = (
        ("Messages", DATASET["messages"]),
        ("Ham", DATASET["ham"]),
        ("Spam", DATASET["spam"]),
        ("Train", DATASET["train"]),
        ("Validation", DATASET["validation"]),
        ("Test", DATASET["test"]),
    )
    for column, (label, value) in zip(counts, labels):
        column.metric(label, f"{value:,}")
    st.caption(
        "UCI SMS Spam Collection combined with the Super SMS Dataset. "
        "Duplicate texts were removed before a stratified 80/10/10 split. "
        "Ham is 61.1% of the messages and spam is 38.9%. Spam is the positive class."
    )
    ham_image = Path(__file__).resolve().parent / "ham_wordcloud.png"
    spam_image = Path(__file__).resolve().parent / "spam_wordcloud.png"
    if ham_image.exists() and spam_image.exists():
        st.markdown("**Word clouds**")
        st.caption("Frequent words in the combined collection, with function words removed.")
        ham_col, spam_col = st.columns(2)
        ham_col.image(str(ham_image), width="stretch")
        spam_col.image(str(spam_image), width="stretch")


def render_test_report(reports):
    payload = reports["splits"]["test"]
    st.subheader("Test set")
    st.caption(
        f"{payload['messages']:,} held-out messages. "
        "Precision, recall, and F1 are for spam. "
        f"The hybrid calls Qwen only when DistilBERT is within {reports['margin']:.2f} of its threshold."
    )
    st.dataframe(metric_frame(payload).style.format("{:.4f}"), width="stretch")
    st.markdown("**Test confusion matrices**")
    st.caption("Rows are the true label. Columns are the prediction.")
    columns = st.columns(3)
    for column, approach in zip(columns, APPROACHES):
        with column:
            st.plotly_chart(
                confusion_figure(payload["approaches"][approach]["confusion"], approach),
                width="stretch",
            )


st.title("SMS Spam Detection")
st.write(
    "A comparison of a TF-IDF logistic regression, a GRU, and DistilBERT, "
    "with Qwen used only when DistilBERT's confidence is weak."
)

render_dataset()

SAMPLES = (
    (
        "Sample: Qwen stays off",
        "Ok lar... Joking wif u oni...",
    ),
    (
        "Sample: Qwen is called",
        "Guess who am I?This is the first time I created a web page WWW.ASJESUS.COM read all I wrote. I'm waiting for your opinions. I want to be your friend 1/1",
    ),
)

st.subheader("Predict a message")
st.caption(
    "Paste your own SMS, or load a sample. Both samples are meant for Hybrid. "
    "The first is a strong ham message, so Qwen stays off. "
    "The second sits close to DistilBERT's threshold, so Qwen is called."
)
if "sms_text" not in st.session_state:
    st.session_state.sms_text = ""
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Hybrid"
sample_cols = st.columns(2)
for column, (label, sample_text) in zip(sample_cols, SAMPLES):
    if column.button(label, width="stretch"):
        st.session_state.sms_text = sample_text
        st.session_state.model_choice = "Hybrid"
option = st.selectbox("Model", OPTIONS, key="model_choice")
st.caption(OPTION_NOTES[option])
message = st.text_area("Text message", height=140, placeholder="Paste an SMS", key="sms_text")
if st.button("Predict", type="primary"):
    if not message.strip():
        st.warning("Paste an SMS first.")
    else:
        logreg, gru, distilbert = load_models()
        if option != "LLM only" and logreg is None and gru is None and distilbert is None:
            st.error("No saved models are in Code/artifacts yet. Run the notebook, then restart this app.")
        else:
            with st.status("Starting the prediction.", expanded=True) as status:
                result = predict(option, message, logreg, gru, distilbert, status)
                status.update(label="Prediction finished.", state="complete")
            render_response(result)

st.divider()
reports = load_reports()
if reports is None or "test" not in reports.get("splits", {}):
    st.info("The test-set scores are not saved yet.")
else:
    render_test_report(reports)
