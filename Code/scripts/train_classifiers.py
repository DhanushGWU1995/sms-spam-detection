#!/usr/bin/env python3
"""Train traditional + DistilBERT classifiers with domain linguistic features.

Traditional models use:
- domain-stopword TF-IDF (word + char)
- POS / NER / scam-signal / GloVe-summary dense features

Advanced model:
- DistilBERT with proper WordPiece tokenization, max-length padding, class weights
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store
from backend.app.nlp.domain_lexicon import domain_tokenizer, extract_linguistic_matrix, linguistic_feature_names

SEED = 42
DISTILBERT_MAX_LEN = 160
DISTILBERT_EPOCHS = 3


def load_split(name: str) -> pd.DataFrame:
    settings = get_settings()
    store = get_store()
    relative = f"{settings.processed_key.rstrip('/')}/messages_{name}.csv"
    try:
        path = store.ensure_local(relative)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing s3 object {store.s3_uri_for(relative)}. Run scripts/prepare_baseline_data.py first."
        ) from exc
    return pd.read_csv(path)


def build_text_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    tokenizer=domain_tokenizer,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=12000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=6000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_feature_matrix(vectorizer: FeatureUnion, texts: list[str], fit: bool = False):
    if fit:
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)
    ling = csr_matrix(extract_linguistic_matrix(texts))
    return hstack([tfidf, ling]).tocsr()


def evaluate(name: str, model, vectorizer, scaler, frame: pd.DataFrame) -> dict:
    texts = frame["text"].astype(str).tolist()
    x = build_feature_matrix(vectorizer, texts, fit=False)
    if scaler is not None:
        x = scaler.transform(x)
    pred = model.predict(x)
    y = frame["label"].astype(str)
    return {
        "model": name,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "report": classification_report(y, pred, zero_division=0, output_dict=True),
    }


def maybe_train_distilbert(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, labels: list[str]
) -> dict | None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:  # pragma: no cover
        print("Skipping DistilBERT:", exc)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Class weights so rare scam subtypes are not ignored.
    counts = train_df["label"].astype(str).value_counts()
    weights = torch.tensor(
        [1.0 / float(counts.get(id2label[i], 1)) for i in range(len(labels))],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)

    class TextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.texts = frame["text"].astype(str).tolist()
            self.y = [label2id[label] for label in frame["label"].astype(str).tolist()]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=DISTILBERT_MAX_LEN,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(self.y[idx], dtype=torch.long)
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    train_loader = DataLoader(TextDataset(train_df), batch_size=16, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df), batch_size=32, shuffle=False)
    epochs = DISTILBERT_EPOCHS
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_state = None
    best_val_f1 = -1.0

    def predict_loader(loader: DataLoader) -> tuple[list[str], list[str]]:
        model.eval()
        preds: list[str] = []
        truth: list[str] = []
        with torch.no_grad():
            for batch in loader:
                y = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                preds.extend(id2label[i] for i in pred_ids)
                truth.extend(id2label[int(i)] for i in y.tolist())
        return truth, preds

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            labels_tensor = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = criterion(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        y_val, val_pred = predict_loader(val_loader)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        print(f"DistilBERT epoch {epoch} loss={np.mean(losses):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = DataLoader(TextDataset(test_df), batch_size=32, shuffle=False)
    y_true, test_pred = predict_loader(test_loader)
    metrics = {
        "model": "distilbert",
        "macro_f1": float(f1_score(y_true, test_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, test_pred, zero_division=0, output_dict=True),
        "val_macro_f1": best_val_f1,
        "max_length": DISTILBERT_MAX_LEN,
        "epochs": epochs,
    }
    settings = get_settings()
    store = get_store()
    out_dir = store.work_dir(f"{settings.models_key.rstrip('/')}/distilbert_scam")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    for uri in store.upload_tree(out_dir, f"{settings.models_key.rstrip('/')}/distilbert_scam"):
        print("wrote", uri)
    return metrics


def main() -> None:
    settings = get_settings()
    store = get_store()
    model_key = settings.models_key.rstrip("/")
    metrics_key = settings.metrics_key.rstrip("/")
    model_dir = store.work_dir(model_key)
    metrics_dir = store.work_dir(metrics_key)
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    labels = sorted(train_df["label"].astype(str).unique().tolist())

    print("Building domain TF-IDF + POS/NER/GloVe-summary features…", flush=True)
    vectorizer = build_text_vectorizer()
    train_texts = train_df["text"].astype(str).tolist()
    x_train = build_feature_matrix(vectorizer, train_texts, fit=True)
    y_train = train_df["label"].astype(str)
    scaler = MaxAbsScaler()
    x_train = scaler.fit_transform(x_train)

    min_class = int(y_train.value_counts().min())
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=4000, class_weight="balanced", random_state=SEED, solver="lbfgs"
        ),
    }
    if min_class >= 2:
        models["linear_svm"] = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=SEED),
            cv=min(3, min_class),
        )
    else:
        print(f"Skipping linear_svm (rarest train class has {min_class} row(s)).")

    metrics_rows = []
    fitted = {}
    for name, model in models.items():
        print("Training", name, flush=True)
        model.fit(x_train, y_train)
        fitted[name] = model
        metrics_rows.append(evaluate(name, model, vectorizer, scaler, test_df))
        print(name, "macro_f1", metrics_rows[-1]["macro_f1"], flush=True)

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "models": fitted,
        "tfidf_only_models": [],
        "labels": labels,
        "linguistic_feature_names": linguistic_feature_names(),
        "feature_note": (
            "TF-IDF with domain SMS stopwords + POS/NER/scam-signal features + "
            "compressed spaCy/GloVe semantic summary. Traditional models: LR + Linear SVM."
        ),
    }
    bundle_path = model_dir / "sklearn_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print("wrote", store.upload_file(bundle_path, f"{model_key}/sklearn_bundle.joblib"))
    print("wrote", store.write_json(f"{model_key}/label_list.json", labels))

    bert_metrics = maybe_train_distilbert(train_df, val_df, test_df, labels)
    if bert_metrics:
        metrics_rows.append(bert_metrics)
        print("distilbert macro_f1", bert_metrics["macro_f1"], flush=True)

    out = {
        "seed": SEED,
        "labels": labels,
        "results": metrics_rows,
        "feature_note": bundle["feature_note"],
        "storage": "s3",
    }
    print("wrote", store.write_json(f"{metrics_key}/classification_metrics.json", out))
    _ = metrics_dir  # work dir created for ephemeral debug copies


if __name__ == "__main__":
    main()
