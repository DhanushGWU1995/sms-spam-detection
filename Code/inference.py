"""Load the three saved SMS models and score one message.

The logistic regression and the GRU see cleaned text. DistilBERT sees the raw message.
"""

import json
from pathlib import Path

import joblib

from preprocess import preprocess_text

def build_gru_classifier():
    from torch import nn

    class _GRUClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden=64, pad_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, tokens, lengths):
            embedded = self.embedding(tokens)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            return self.fc(hidden[-1]).squeeze(-1)

    return _GRUClassifier


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.pkl"
LOGREG_PATH = ARTIFACT_DIR / "logreg_model.pkl"
GRU_PATH = ARTIFACT_DIR / "gru.pt"
BERT_DIR = ARTIFACT_DIR / "distilbert"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"


def load_thresholds():
    if not THRESHOLD_PATH.exists():
        return {"logreg": 0.5, "gru": 0.5, "distilbert": 0.5}
    with THRESHOLD_PATH.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    return {
        "logreg": float(saved.get("logreg", 0.5)),
        "gru": float(saved.get("gru", 0.5)),
        "distilbert": float(saved.get("distilbert", 0.5)),
    }


def load_logreg():
    if not (VECTORIZER_PATH.exists() and LOGREG_PATH.exists()):
        return None
    return {
        "vectorizer": joblib.load(VECTORIZER_PATH),
        "model": joblib.load(LOGREG_PATH),
        "threshold": load_thresholds()["logreg"],
    }


def score_logreg(bundle, text):
    features = bundle["vectorizer"].transform([preprocess_text(text)])
    probability = float(bundle["model"].predict_proba(features)[0, 1])
    return _result("Logistic Regression", "TF-IDF baseline", probability, bundle["threshold"])


def load_gru():
    if not GRU_PATH.exists():
        return None
    import torch

    checkpoint = torch.load(GRU_PATH, map_location="cpu", weights_only=False)
    stoi = checkpoint["stoi"]
    model = build_gru_classifier()(
        len(stoi),
        embed_dim=checkpoint.get("embed_dim", 64),
        hidden=checkpoint.get("hidden", 64),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "stoi": stoi, "threshold": load_thresholds()["gru"]}


def score_gru(bundle, text):
    import torch

    ids = [bundle["stoi"].get(tok, 1) for tok in preprocess_text(text).split()][:60]
    if not ids:
        ids = [1]
    tokens = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    with torch.no_grad():
        probability = float(torch.sigmoid(bundle["model"](tokens, lengths)).item())
    return _result("GRU", "Recurrent network", probability, bundle["threshold"])


def load_distilbert():
    if not (BERT_DIR / "config.json").exists():
        return None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "threshold": load_thresholds()["distilbert"],
    }


def score_distilbert(bundle, text):
    import torch

    encoded = bundle["tokenizer"](
        text, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        logits = bundle["model"](**encoded).logits
        probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
    return _result("DistilBERT", "Transformer", probability, bundle["threshold"])


def score_message(text, logreg, gru, distilbert):
    scored = []
    if logreg is not None:
        scored.append(score_logreg(logreg, text))
    else:
        scored.append(_missing("Logistic Regression", "TF-IDF baseline"))
    if gru is not None:
        scored.append(score_gru(gru, text))
    else:
        scored.append(_missing("GRU", "Recurrent network"))
    if distilbert is not None:
        scored.append(score_distilbert(distilbert, text))
    else:
        scored.append(_missing("DistilBERT", "Transformer"))
    return scored


def _result(name, role, probability, threshold):
    return {
        "name": name,
        "role": role,
        "probability": probability,
        "threshold": threshold,
        "spam": probability >= threshold,
        "ready": True,
    }


def _missing(name, role):
    return {
        "name": name,
        "role": role,
        "ready": False,
        "detail": "Run Code/main.ipynb so this model is saved, then restart the app.",
    }
