"""Score an LLM only on the SMS messages the three trained models are unsure about.

The margin is chosen on the validation split. The test split is scored once.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from inference import (
    ARTIFACT_DIR,
    load_distilbert,
    load_gru,
    load_logreg,
    load_thresholds,
)
from preprocess import preprocess_text

LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)


def _metrics(y_true, pred):
    pred = np.asarray(pred).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def _gru_proba(bundle, texts):
    import torch

    model = bundle["model"]
    stoi = bundle["stoi"]
    encoded = []
    for text in texts:
        ids = [stoi.get(tok, 1) for tok in preprocess_text(text).split()][:60]
        encoded.append(ids or [1])
    lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long)
    padded = torch.zeros(len(encoded), int(lengths.max()), dtype=torch.long)
    for i, seq in enumerate(encoded):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(padded, lengths)
    return torch.sigmoid(logits).numpy()


def _bert_proba(bundle, texts, batch_size=64):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle["model"].to(device)
    tokenizer = bundle["tokenizer"]
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch, truncation=True, padding=True, max_length=128, return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probs = torch.softmax(model(**encoded).logits, dim=-1)[:, 1]
            chunks.append(probs.cpu().numpy())
    return np.concatenate(chunks)


def _logreg_proba(bundle, texts):
    cleaned = [preprocess_text(text) for text in texts]
    return bundle["model"].predict_proba(bundle["vectorizer"].transform(cleaned))[:, 1]


def _prompt(tokenizer, text):
    messages = [
        {
            "role": "system",
            "content": "Classify the SMS as spam or ham. Reply with one word: spam or ham.",
        },
        {"role": "user", "content": str(text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse_label(text):
    word = text.strip().lower().split()
    if not word:
        return -1
    token = word[0].strip(".,:;")
    if token.startswith("spam"):
        return 1
    if token.startswith("ham"):
        return 0
    return -1


def load_qwen():
    return _load_llm()


def label_message(text, bundle):
    """Ask Qwen for one SMS. bundle is the tuple returned by load_qwen."""
    labels, cost = _llm_predict([text], *bundle, batch_size=1)
    parsed = int(labels[0]) >= 0
    return {
        "name": "Qwen",
        "role": "Open LLM, called when DistilBERT is close to its threshold",
        "ready": parsed,
        "spam": bool(int(labels[0]) == 1) if parsed else None,
        "seconds": cost["seconds"],
        "detail": "" if parsed else "Qwen did not answer spam or ham. We keep DistilBERT's label.",
    }


def saved_weak_margin(default=0.15):
    path = ARTIFACT_DIR / "llm_summary.json"
    if not path.exists():
        return default
    saved = json.loads(path.read_text(encoding="utf-8"))
    return float(saved.get("margin", default))


def final_decision(outcomes, weak=False, qwen_spam=None):
    """Use the trained models, or Qwen when DistilBERT's confidence is weak."""
    ready = [item for item in outcomes if item.get("ready")]
    if weak:
        return {"called_qwen": True, "spam": qwen_spam}
    if not ready:
        return {"called_qwen": False, "spam": None}
    votes = sum(1 for item in ready if item["spam"])
    return {"called_qwen": False, "spam": votes >= (len(ready) / 2)}


def _load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, model, device


def _llm_predict(texts, tokenizer, model, device, batch_size=32):
    import torch

    labels = np.full(len(texts), -1, dtype=int)
    input_tokens = 0
    output_tokens = 0
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        prompts = [_prompt(tokenizer, text) for text in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_len = encoded["attention_mask"].sum(dim=1)
        input_tokens += int(input_len.sum().item())
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        output_tokens += int((new_tokens != tokenizer.pad_token_id).sum().item())
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for offset, reply in enumerate(decoded):
            labels[start + offset] = _parse_label(reply)
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % 512 == 0:
            print(f"LLM scored {done}/{len(texts)}", flush=True)
    elapsed = time.perf_counter() - started
    return labels, {
        "model": LLM_NAME,
        "device": str(device),
        "seconds": elapsed,
        "messages": len(texts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds_per_message": elapsed / max(len(texts), 1),
    }


def _weak_mask(bert_proba, threshold, margin):
    return np.abs(bert_proba - threshold) <= margin


def _majority(preds):
    votes = preds["logreg"] + preds["gru"] + preds["distilbert"]
    return (votes >= 2).astype(int)


def _hybrid(bert_pred, llm_pred, route):
    pred = np.array(bert_pred, copy=True)
    use = route & (llm_pred >= 0)
    pred[use] = llm_pred[use]
    return pred


def _choose_margin(y_true, bert_proba, traditional_pred, llm_pred, threshold):
    best_margin, best_f1 = MARGINS[0], -1.0
    rows = []
    for margin in MARGINS:
        route = _weak_mask(bert_proba, threshold, margin)
        pred = _hybrid(traditional_pred, llm_pred, route)
        scores = _metrics(y_true, pred)
        scores["margin"] = margin
        scores["routed"] = float(route.mean())
        rows.append(scores)
        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            best_margin = margin
    return best_margin, rows


def run_llm_evaluation(artifact_dir=ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    val_df = pd.read_csv(artifact_dir / "val.csv")
    test_df = pd.read_csv(artifact_dir / "test.csv")
    thresholds = load_thresholds()
    logreg = load_logreg()
    gru = load_gru()
    bert = load_distilbert()
    if logreg is None or gru is None or bert is None:
        raise FileNotFoundError("Train the three models before running the LLM cascade.")

    def pack(frame):
        texts = frame["Text"].astype(str).tolist()
        labels = frame["label"].to_numpy().astype(int)
        proba = {
            "logreg": _logreg_proba(logreg, texts),
            "gru": _gru_proba(gru, texts),
            "distilbert": _bert_proba(bert, texts),
        }
        preds = {
            name: (proba[name] >= thresholds[name]).astype(int)
            for name in proba
        }
        return texts, labels, proba, preds

    print("Scoring validation and test with the three saved models.", flush=True)
    val_texts, y_val, val_proba, val_preds = pack(val_df)
    test_texts, y_test, test_proba, test_preds = pack(test_df)

    print("Running the LLM on validation and test.", flush=True)
    tokenizer, llm, device = _load_llm()
    val_llm, val_cost = _llm_predict(val_texts, tokenizer, llm, device)
    test_llm, test_cost = _llm_predict(test_texts, tokenizer, llm, device)
    llm_cost = {
        "model": LLM_NAME,
        "device": str(device),
        "seconds": val_cost["seconds"] + test_cost["seconds"],
        "messages": val_cost["messages"] + test_cost["messages"],
        "input_tokens": val_cost["input_tokens"] + test_cost["input_tokens"],
        "output_tokens": val_cost["output_tokens"] + test_cost["output_tokens"],
    }
    llm_cost["seconds_per_message"] = llm_cost["seconds"] / max(llm_cost["messages"], 1)
    val_traditional = _majority(val_preds)
    test_traditional = _majority(test_preds)
    margin, val_rows = _choose_margin(
        y_val,
        val_proba["distilbert"],
        val_traditional,
        val_llm,
        thresholds["distilbert"],
    )

    route = _weak_mask(test_proba["distilbert"], thresholds["distilbert"], margin)
    hybrid = _hybrid(test_traditional, test_llm, route)
    llm_all = np.where(test_llm >= 0, test_llm, 0)

    traditional_row = _metrics(y_test, test_traditional)
    traditional_row["approach"] = "Traditional models"
    traditional_row["routed_to_llm"] = 0.0
    llm_row = _metrics(y_test, llm_all)
    llm_row["approach"] = "LLM only"
    llm_row["routed_to_llm"] = 1.0
    hybrid_row = _metrics(y_test, hybrid)
    hybrid_row["approach"] = "Hybrid"
    hybrid_row["routed_to_llm"] = float(route.mean())
    table = [traditional_row, llm_row, hybrid_row]

    asked = route & (test_llm >= 0)
    changed = asked & (hybrid != test_traditional)
    helped = changed & (hybrid == y_test)
    hurt = changed & (hybrid != y_test)
    summary = {
        "llm": llm_cost,
        "margin": margin,
        "rule": (
            "Qwen runs when the DistilBERT spam probability is within the chosen "
            "margin of its threshold. Every other message keeps the majority vote "
            "of logistic regression, the GRU, and DistilBERT."
        ),
        "validation_margins": val_rows,
        "test_messages": int(len(y_test)),
        "routed_messages": int(route.sum()),
        "unparsed_test": int((test_llm < 0).sum()),
        "flips_correct": int(helped.sum()),
        "flips_wrong": int(hurt.sum()),
        "traditional_f1": traditional_row["f1"],
        "hybrid_f1": hybrid_row["f1"],
        "llm_f1": llm_row["f1"],
    }
    frame = pd.DataFrame(table).set_index("approach")
    frame = frame[["f1", "precision", "recall", "accuracy", "routed_to_llm"]]
    frame.to_csv(artifact_dir / "llm_metrics.csv")
    (artifact_dir / "llm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(float_format=lambda value: f"{value:.4f}"))
    print(
        f"Weak-confidence margin chosen on validation: {margin:.2f}. "
        f"Test messages sent to Qwen: {int(route.sum())}/{len(y_test)}. "
        f"Correct flips {int(helped.sum())}, wrong flips {int(hurt.sum())}."
    )
    return {"table": frame, "summary": summary}


def _report_row(y_true, pred, routed_rate):
    row = _metrics(y_true, pred)
    matrix = confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(pred).astype(int),
        labels=[0, 1],
    )
    row["confusion"] = matrix.astype(int).tolist()
    row["routed_to_llm"] = float(routed_rate)
    row["messages"] = int(len(y_true))
    return row


def export_split_reports(artifact_dir=ARTIFACT_DIR):
    """Save train and test scores plus confusion matrices for the three approaches.

    The weak-confidence margin is the one already chosen on validation.
    """
    artifact_dir = Path(artifact_dir)
    margin = saved_weak_margin()
    thresholds = load_thresholds()
    logreg = load_logreg()
    gru = load_gru()
    bert = load_distilbert()
    if logreg is None or gru is None or bert is None:
        raise FileNotFoundError("Train the three models before exporting the reports.")

    print("Loading Qwen.", flush=True)
    tokenizer, llm, device = _load_llm()
    splits = {}
    for name in ("train", "test"):
        frame = pd.read_csv(artifact_dir / f"{name}.csv")
        texts = frame["Text"].astype(str).tolist()
        y_true = frame["label"].to_numpy().astype(int)
        print(f"Scoring {name} with the three saved models ({len(texts)} messages).", flush=True)
        proba = {
            "logreg": _logreg_proba(logreg, texts),
            "gru": _gru_proba(gru, texts),
            "distilbert": _bert_proba(bert, texts),
        }
        preds = {key: (proba[key] >= thresholds[key]).astype(int) for key in proba}
        traditional = _majority(preds)
        print(f"Running the LLM on {name}.", flush=True)
        llm_labels, _cost = _llm_predict(texts, tokenizer, llm, device)
        route = _weak_mask(proba["distilbert"], thresholds["distilbert"], margin)
        hybrid = _hybrid(traditional, llm_labels, route)
        llm_all = np.where(llm_labels >= 0, llm_labels, 0)
        splits[name] = {
            "messages": int(len(y_true)),
            "approaches": {
                "Traditional models": _report_row(y_true, traditional, 0.0),
                "LLM only": _report_row(y_true, llm_all, 1.0),
                "Hybrid": _report_row(y_true, hybrid, float(route.mean())),
            },
        }

    report = {
        "margin": margin,
        "positive_class": "spam",
        "order": ["ham", "spam"],
        "splits": splits,
    }
    path = artifact_dir / "approach_reports.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote approach reports to {path}", flush=True)
    for name, payload in splits.items():
        print(name, flush=True)
        for approach, row in payload["approaches"].items():
            print(
                f"  {approach}: f1={row['f1']:.4f} precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f} accuracy={row['accuracy']:.4f} "
                f"confusion={row['confusion']}",
                flush=True,
            )
    return report


if __name__ == "__main__":
    import sys

    if "--reports" in sys.argv:
        export_split_reports()
    else:
        run_llm_evaluation()
