#!/usr/bin/env python3
"""Compare classification approaches on the same held-out splits.

Approaches
----------
1. Traditional NLP — TF-IDF (domain stopwords) + POS/NER/GloVe-summary → LR/SVM vote
2. Advanced NLP — DistilBERT (order-aware transformer)
3. Hybrid NLP — cascade Traditional → Advanced:
   keep Traditional when confident; escalate weak Traditional cases to DistilBERT

Cascade
-------
Traditional NLP → (weak) Advanced NLP / DistilBERT
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store
from backend.app.nlp.domain_lexicon import extract_linguistic_matrix

MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)
DISTILBERT_MAX_LEN = 160


def load_split(name: str) -> pd.DataFrame:
    settings = get_settings()
    store = get_store()
    relative = f"{settings.processed_key.rstrip('/')}/messages_{name}.csv"
    path = store.ensure_local(relative)
    return pd.read_csv(path)


def load_labels() -> list[str]:
    settings = get_settings()
    store = get_store()
    relative = f"{settings.models_key.rstrip('/')}/label_list.json"
    if store.exists(relative):
        return store.read_json(relative)
    return sorted(load_split("train")["label"].astype(str).unique().tolist())


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    y_true_a = np.asarray(y_true, dtype=object)
    y_pred_a = np.asarray(y_pred, dtype=object)
    matrix = confusion_matrix(y_true_a, y_pred_a, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "macro_f1": float(f1_score(y_true_a, y_pred_a, average="macro", zero_division=0, labels=labels)),
        "weighted_f1": float(
            f1_score(y_true_a, y_pred_a, average="weighted", zero_division=0, labels=labels)
        ),
        "macro_precision": float(
            precision_score(y_true_a, y_pred_a, average="macro", zero_division=0, labels=labels)
        ),
        "macro_recall": float(
            recall_score(y_true_a, y_pred_a, average="macro", zero_division=0, labels=labels)
        ),
        "confusion": matrix.astype(int).tolist(),
        "report": classification_report(
            y_true_a, y_pred_a, labels=labels, zero_division=0, output_dict=True
        ),
        "messages": int(len(y_true_a)),
    }


def _majority_vote(pred_lists: list[list[str]], tie_break: list[str]) -> list[str]:
    out: list[str] = []
    for i in range(len(tie_break)):
        votes = Counter(preds[i] for preds in pred_lists)
        best = votes.most_common()
        top_count = best[0][1]
        tied = {label for label, count in best if count == top_count}
        if tie_break[i] in tied:
            out.append(tie_break[i])
        else:
            out.append(sorted(tied)[0])
    return out


def _build_features(bundle: dict[str, Any], texts: list[str]):
    vectorizer = bundle["vectorizer"]
    tfidf = vectorizer.transform(texts)
    if "scaler" in bundle or "linguistic_feature_names" in bundle:
        ling = csr_matrix(extract_linguistic_matrix(texts))
        features = hstack([tfidf, ling]).tocsr()
        scaler = bundle.get("scaler")
        if scaler is not None:
            features = scaler.transform(features)
        return features
    return tfidf


def _proba_margin_stats(probs: np.ndarray, classes: list[str]) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    confidences: list[float] = []
    margins: list[float] = []
    for row in probs:
        order = np.argsort(row)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        labels.append(str(classes[top1]))
        confidences.append(float(row[top1]))
        margins.append(float(row[top1] - row[top2]))
    return labels, np.column_stack([confidences, margins])


def _sklearn_predict(bundle: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    features = _build_features(bundle, texts)
    skip = {"multinomial_nb"}
    preds: dict[str, list[str]] = {}
    margins: dict[str, np.ndarray] = {}
    for name, model in bundle["models"].items():
        if name in skip:
            continue
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)
            classes = [str(c) for c in model.classes_]
            labels, stats = _proba_margin_stats(np.asarray(probs), classes)
            preds[name] = labels
            margins[name] = stats[:, 1]
        else:
            preds[name] = [str(v) for v in model.predict(features)]
    return {"preds": preds, "margins": margins}


def _distilbert_predict(model, tokenizer, device, texts: list[str], batch_size: int = 32):
    import torch

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels: list[str] = []
    confidences: list[float] = []
    margins: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=DISTILBERT_MAX_LEN,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
            for row in probs:
                order = np.argsort(row)[::-1]
                top1 = int(order[0])
                top2 = int(order[1]) if len(order) > 1 else top1
                labels.append(id2label[top1])
                confidences.append(float(row[top1]))
                margins.append(float(row[top1] - row[top2]))
    return labels, np.column_stack([confidences, margins])


def _traditional_weak_mask(
    traditional: list[str],
    lr_labels: list[str] | None,
    svm_labels: list[str] | None,
    trad_margins: np.ndarray,
    margin: float,
) -> np.ndarray:
    weak = np.asarray(trad_margins, dtype=float) <= float(margin)
    if lr_labels is not None and svm_labels is not None:
        weak = weak | (np.asarray(lr_labels, dtype=object) != np.asarray(svm_labels, dtype=object))
    _ = traditional
    return weak


def _hybrid_nlp(
    traditional: list[str],
    advanced: list[str],
    trad_weak: np.ndarray,
) -> list[str]:
    return [advanced[i] if bool(trad_weak[i]) else traditional[i] for i in range(len(traditional))]


def _choose_hybrid_margin(
    y_true: list[str],
    traditional: list[str],
    advanced: list[str],
    lr_labels: list[str] | None,
    svm_labels: list[str] | None,
    trad_margins: np.ndarray,
    labels: list[str],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_margin, best_f1 = MARGINS[0], -1.0
    for margin in MARGINS:
        trad_weak = _traditional_weak_mask(
            traditional, lr_labels, svm_labels, trad_margins, margin
        )
        pred = _hybrid_nlp(traditional, advanced, trad_weak)
        scores = _metrics(y_true, pred, labels)
        rows.append(
            {
                "margin": margin,
                "macro_f1": scores["macro_f1"],
                "routed_to_distilbert": float(np.mean(trad_weak)),
            }
        )
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_margin = margin
    return best_margin, rows


def _pack_approaches(
    y_true: list[str],
    traditional: list[str],
    advanced: list[str],
    hybrid_nlp: list[str],
    labels: list[str],
) -> dict[str, Any]:
    return {
        "Traditional NLP": {
            **_metrics(y_true, traditional, labels),
            "routed_to_advanced": 0.0,
            "description": (
                "Domain-stopword TF-IDF + POS/NER/scam signals + GloVe summary; "
                "majority of LR / SVM."
            ),
        },
        "Advanced NLP": {
            **_metrics(y_true, advanced, labels),
            "routed_to_advanced": 1.0,
            "description": (
                "DistilBERT with WordPiece tokenization, max-length padding, "
                "class-weighted fine-tuning (order-aware)."
            ),
        },
        "Hybrid NLP": {
            **_metrics(y_true, hybrid_nlp, labels),
            "routed_to_advanced": None,
            "description": (
                "Cascade Traditional → Advanced: keep Traditional when confident; "
                "escalate weak Traditional cases to DistilBERT (val-tuned margin)."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ScamShield classification approaches.")
    parser.parse_args()

    settings = get_settings()
    store = get_store()
    processed_key = settings.processed_key.rstrip("/")
    model_key = settings.models_key.rstrip("/")
    metrics_key = settings.metrics_key.rstrip("/")
    metrics_dir = store.work_dir(metrics_key)

    labels = load_labels()
    split_meta = store.read_json(f"{processed_key}/split_meta.json") if store.exists(f"{processed_key}/split_meta.json") else {}
    bundle_rel = f"{model_key}/sklearn_bundle.joblib"
    bert_rel = f"{model_key}/distilbert_scam"
    if not store.exists(bundle_rel):
        raise FileNotFoundError(f"Missing {store.s3_uri_for(bundle_rel)}. Run scripts/train_classifiers.py first.")
    if not store.exists(f"{bert_rel}/config.json"):
        raise FileNotFoundError(f"Missing DistilBERT at {store.s3_uri_for(bert_rel)}.")
    bundle_path = store.ensure_local(bundle_rel)
    bert_dir = store.ensure_prefix_local(bert_rel)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bundle = joblib.load(bundle_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    bert = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)

    def score_frame(frame: pd.DataFrame) -> dict[str, Any]:
        texts = frame["text"].astype(str).tolist()
        sk_out = _sklearn_predict(bundle, texts)
        sk = sk_out["preds"]
        sk_margins = sk_out["margins"]
        adv, bert_stats = _distilbert_predict(bert, tokenizer, device, texts)
        y_true = frame["label"].astype(str).tolist()
        traditional_lists = [sk[name] for name in sorted(sk)]
        tie = sk.get("logistic_regression") or traditional_lists[0]
        traditional = _majority_vote(traditional_lists, tie)
        lr_labels = sk.get("logistic_regression")
        svm_labels = sk.get("linear_svm")
        margin_parts = []
        if "linear_svm" in sk_margins:
            margin_parts.append(sk_margins["linear_svm"])
        if "logistic_regression" in sk_margins:
            margin_parts.append(sk_margins["logistic_regression"])
        if not margin_parts:
            raise RuntimeError("Need LR/SVM predict_proba margins for Traditional→Advanced cascade.")
        trad_margins = margin_parts[0]
        for part in margin_parts[1:]:
            trad_margins = np.minimum(trad_margins, part)
        return {
            "texts": texts,
            "y_true": y_true,
            "traditional": traditional,
            "advanced": adv,
            "lr_labels": lr_labels,
            "svm_labels": svm_labels,
            "trad_margin": trad_margins,
            "top_conf": bert_stats[:, 0],
            "top_margin": bert_stats[:, 1],
        }

    def apply_hybrid(pack: dict[str, Any], hybrid_margin: float) -> dict[str, Any]:
        trad_weak = _traditional_weak_mask(
            pack["traditional"],
            pack.get("lr_labels"),
            pack.get("svm_labels"),
            pack["trad_margin"],
            hybrid_margin,
        )
        hybrid_nlp = _hybrid_nlp(pack["traditional"], pack["advanced"], trad_weak)
        out = dict(pack)
        out["trad_weak"] = trad_weak
        out["hybrid_nlp"] = hybrid_nlp
        out["routed_to_distilbert"] = float(np.mean(trad_weak))
        out["hybrid_margin"] = hybrid_margin
        return out

    print("Scoring validation and test with Traditional + DistilBERT models.", flush=True)
    val_raw = score_frame(load_split("val"))
    test_raw = score_frame(load_split("test"))
    hybrid_margin, hybrid_margin_rows = _choose_hybrid_margin(
        val_raw["y_true"],
        val_raw["traditional"],
        val_raw["advanced"],
        val_raw.get("lr_labels"),
        val_raw.get("svm_labels"),
        val_raw["trad_margin"],
        labels,
    )
    print(f"Validation chose Traditional→Advanced margin={hybrid_margin}", flush=True)
    for row in hybrid_margin_rows:
        print(
            f"  margin={row['margin']:.2f} val_hybrid_macro_f1={row['macro_f1']:.4f} "
            f"routed_to_distilbert={row['routed_to_distilbert']:.4f}",
            flush=True,
        )
    val_pack = apply_hybrid(val_raw, hybrid_margin)
    test_pack = apply_hybrid(test_raw, hybrid_margin)

    splits: dict[str, Any] = {}
    for name, pack in (("val", val_pack), ("test", test_pack)):
        approaches = _pack_approaches(
            pack["y_true"],
            pack["traditional"],
            pack["advanced"],
            pack["hybrid_nlp"],
            labels,
        )
        approaches["Hybrid NLP"]["routed_to_advanced"] = float(pack["routed_to_distilbert"])
        splits[name] = {"messages": int(len(pack["y_true"])), "approaches": approaches}

    order = ["Traditional NLP", "Advanced NLP", "Hybrid NLP"]
    summary = {
        "primary_metric": "macro_f1",
        "labels": labels,
        "order": order,
        "dataset_name": split_meta.get(
            "dataset_name", "UCI SMS Spam Collection + Super SMS Dataset"
        ),
        "datasets_used": split_meta.get("datasets_used", []),
        "comparison_split": "test",
        "rule": (
            "Cascade Traditional → Advanced. Hybrid NLP keeps Traditional when confident "
            "and escalates weak Traditional cases to DistilBERT."
        ),
        "hybrid_margin": hybrid_margin,
        "hybrid_validation_margins": hybrid_margin_rows,
        "label_note": (
            "Binary ham/spam from UCI + Super SMS; multi-class spam subtypes are silver labels."
        ),
        "test_messages": splits["test"]["messages"],
        "routed_to_distilbert": test_pack["routed_to_distilbert"],
        "split_meta": {
            "rows": split_meta.get("rows"),
            "train": split_meta.get("train"),
            "val": split_meta.get("val"),
            "test": split_meta.get("test"),
            "binary_counts": split_meta.get("binary_counts"),
            "merge": split_meta.get("merge"),
        },
    }
    for key in order:
        summary[f"{key.lower().replace(' ', '_')}_macro_f1"] = splits["test"]["approaches"][key][
            "macro_f1"
        ]

    report = {
        "hybrid_margin": hybrid_margin,
        "labels": labels,
        "order": order,
        "primary_metric": "macro_f1",
        "dataset_name": summary["dataset_name"],
        "datasets_used": summary["datasets_used"],
        "comparison_split": "test",
        "hybrid_rule": (
            "Traditional → Advanced: keep Traditional when confident; escalate when "
            "Traditional top-margin is weak or LR/SVM disagree. Margin maximized on validation "
            "for Hybrid NLP macro F1."
        ),
        "splits": splits,
        "summary": summary,
    }

    table_rows = []
    for name in order:
        row = splits["test"]["approaches"][name]
        table_rows.append(
            {
                "approach": name,
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
                "macro_precision": row["macro_precision"],
                "macro_recall": row["macro_recall"],
                "accuracy": row["accuracy"],
                "routed_to_advanced": row.get("routed_to_advanced", 0.0),
            }
        )
    frame = pd.DataFrame(table_rows).set_index("approach")
    print(frame.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)

    hybrid_test = splits["test"]["approaches"]["Hybrid NLP"]
    print(
        f"Hybrid NLP (Traditional→Advanced) macro_f1={hybrid_test['macro_f1']:.4f} "
        f"acc={hybrid_test['accuracy']:.4f} "
        f"trad_margin={hybrid_margin:.2f} "
        f"routed_to_distilbert={test_pack.get('routed_to_distilbert', 0):.4f}",
        flush=True,
    )

    report["storage"] = "s3"
    summary["storage"] = "s3"
    reports_path = metrics_dir / "approach_reports.json"
    summary_path = metrics_dir / "approach_summary.json"
    csv_path = metrics_dir / "approach_metrics.csv"
    reports_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.to_csv(csv_path)
    print("wrote", store.upload_file(reports_path, f"{metrics_key}/approach_reports.json"), flush=True)
    print("wrote", store.upload_file(summary_path, f"{metrics_key}/approach_summary.json"), flush=True)
    print("wrote", store.upload_file(csv_path, f"{metrics_key}/approach_metrics.csv"), flush=True)


if __name__ == "__main__":
    main()
