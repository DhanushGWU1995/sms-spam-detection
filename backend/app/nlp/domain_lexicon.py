"""Domain-specific SMS / consumer-fraud lexicons and linguistic helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "this", "that", "these",
    "those", "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "their", "so", "than", "too", "very", "just", "about",
}

SMS_NOISE_STOPWORDS = {
    "lol", "lmao", "haha", "hahaha", "ok", "okay", "ya", "yah", "yup", "nah",
    "uhm", "umm", "hmm", "pls", "plz", "thx", "thanks", "thank", "hi", "hey",
    "hello", "dear", "sir", "madam", "regards",
}

# Keep scam-critical tokens even if they look like stopwords.
SCAM_KEEP_TOKENS = {
    "free", "win", "won", "winner", "urgent", "immediately", "verify", "account",
    "password", "bank", "irs", "ssn", "otp", "code", "click", "link", "http",
    "https", "www", "call", "text", "msg", "prize", "claim", "refund", "suspend",
    "suspended", "locked", "confirm", "update", "security", "alert", "payment",
    "gift", "card", "wire", "bitcoin", "crypto", "zelle", "venmo", "paypal",
}

DOMAIN_STOPWORDS = frozenset((ENGLISH_STOPWORDS | SMS_NOISE_STOPWORDS) - SCAM_KEEP_TOKENS)

POS_FEATURE_TAGS = ("NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "PRON")
NER_FEATURE_LABELS = ("PERSON", "ORG", "GPE", "MONEY", "DATE", "CARDINAL", "PERCENT")
GLOVE_SUMMARY_DIM = 8


@lru_cache(maxsize=1)
def load_spacy():
    try:
        import spacy

        for name in ("en_core_web_md", "en_core_web_sm"):
            try:
                return spacy.load(name, disable=["lemmatizer"])
            except Exception:
                continue
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None


def domain_tokenizer(text: str) -> list[str]:
    import re

    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|https?|www", (text or "").lower())
    return [tok for tok in tokens if tok not in DOMAIN_STOPWORDS and len(tok) > 1]


def linguistic_feature_names() -> list[str]:
    names = [
        "n_tokens",
        "n_chars",
        "avg_token_len",
        "pct_digits",
        "pct_upper",
        "has_url",
        "has_email",
        "has_phone",
        "n_urgency",
        "n_credential",
        "n_payment",
        "n_impersonation",
        "asks_sensitive",
        "exclaim_count",
        "question_count",
    ]
    names.extend(f"pos_{tag.lower()}" for tag in POS_FEATURE_TAGS)
    names.extend(f"ner_{label.lower()}" for label in NER_FEATURE_LABELS)
    names.append("has_glove_vector")
    names.extend(f"glove_{i}" for i in range(GLOVE_SUMMARY_DIM))
    return names


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def extract_linguistic_vector(text: str, signals: dict[str, Any] | None = None) -> np.ndarray:
    """POS / NER / domain-signal vector used by traditional + hybrid models."""
    from backend.app.nlp.features import extract_signals

    text = text or ""
    sig = signals or extract_signals(text).to_dict()
    tokens = domain_tokenizer(text)
    n_tokens = max(len(tokens), 1)
    n_chars = max(len(text), 1)
    digit_chars = sum(ch.isdigit() for ch in text)
    upper_chars = sum(ch.isupper() for ch in text)

    base = [
        float(len(tokens)),
        float(len(text)),
        float(np.mean([len(t) for t in tokens]) if tokens else 0.0),
        _safe_ratio(digit_chars, n_chars),
        _safe_ratio(upper_chars, n_chars),
        1.0 if sig.get("urls") else 0.0,
        1.0 if sig.get("emails") else 0.0,
        1.0 if sig.get("phones") else 0.0,
        float(len(sig.get("urgency_cues") or [])),
        float(len(sig.get("credential_requests") or [])),
        float(len(sig.get("payment_methods") or [])),
        float(len(sig.get("impersonated_orgs") or [])),
        1.0 if sig.get("asks_for_sensitive_info") else 0.0,
        float(text.count("!")),
        float(text.count("?")),
    ]

    pos_counts = {tag: 0.0 for tag in POS_FEATURE_TAGS}
    ner_counts = {label: 0.0 for label in NER_FEATURE_LABELS}
    glove_summary = np.zeros(GLOVE_SUMMARY_DIM, dtype=np.float32)
    has_glove = 0.0

    nlp = load_spacy()
    if nlp is not None:
        doc = nlp(text[:1000])
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1.0
        for ent in getattr(doc, "ents", []):
            if ent.label_ in ner_counts:
                ner_counts[ent.label_] += 1.0
        # spaCy md vectors are GloVe-trained; compress to a small semantic summary.
        if getattr(doc, "has_vector", False) and float(getattr(doc, "vector_norm", 0.0) or 0.0) > 0:
            vec = np.asarray(doc.vector, dtype=np.float32)
            has_glove = 1.0
            bucket = max(len(vec) // GLOVE_SUMMARY_DIM, 1)
            for i in range(GLOVE_SUMMARY_DIM):
                chunk = vec[i * bucket : (i + 1) * bucket]
                glove_summary[i] = float(chunk.mean()) if len(chunk) else 0.0

    pos_feats = [_safe_ratio(pos_counts[tag], n_tokens) for tag in POS_FEATURE_TAGS]
    ner_feats = [float(ner_counts[label]) for label in NER_FEATURE_LABELS]
    return np.asarray(
        base + pos_feats + ner_feats + [has_glove] + glove_summary.tolist(),
        dtype=np.float32,
    )


def extract_linguistic_matrix(texts: list[str], batch_size: int = 256) -> np.ndarray:
    rows = [extract_linguistic_vector(text) for text in texts]
    return np.vstack(rows) if rows else np.zeros((0, len(linguistic_feature_names())), dtype=np.float32)
