#!/usr/bin/env python3
"""Build training splits from the same two SMS sources as the previous project.

Sources
-------
1. UCI SMS Spam Collection — Category/Text (ham/spam)
2. Super SMS Dataset — SMSes/Labels (0=ham, 1=spam)

We prefer the already-merged `sms_combined.csv` when present. Otherwise we rebuild
the merge: same columns, drop duplicate texts, and keep the UCI label on conflicts.
UCI/Super only provide binary labels. Multi-class subtypes on spam rows are silver
labels from heuristics and are clearly marked in metadata.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store
from backend.app.nlp.features import extract_signals

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

CODE_CANDIDATES = [
    ROOT / "archive" / "sms_spam_v1" / "Code",
    ROOT / "Code",
]


def _find(*names: str) -> Path | None:
    for folder in CODE_CANDIDATES:
        for name in names:
            path = folder / name
            if path.exists():
                return path
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    """Read SMS CSVs robustly (Super SMS includes non-UTF8 bytes)."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def _normalize_binary(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "ham": "ham",
                "spam": "spam",
                "0": "ham",
                "1": "spam",
                "0.0": "ham",
                "1.0": "spam",
            }
        )
    )


def load_uci() -> pd.DataFrame:
    csv_path = _find("sms_spam.csv")
    if csv_path is not None:
        frame = _read_csv(csv_path)
        frame = frame.rename(columns={"Category": "binary_label", "Text": "text"})
        frame["binary_label"] = _normalize_binary(frame["binary_label"])
        frame = frame.dropna(subset=["text", "binary_label"])
        frame["source"] = "uci_sms_spam"
        return frame[["text", "binary_label", "source"]].reset_index(drop=True)

    zip_path = ROOT / "archive" / "sms_spam_v1" / "sms+spam+collection.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
            with archive.open(member) as handle:
                frame = pd.read_csv(handle, sep="\t", header=None, names=["binary_label", "text"])
        frame["binary_label"] = _normalize_binary(frame["binary_label"])
        frame["source"] = "uci_sms_spam"
        return frame[["text", "binary_label", "source"]].dropna().reset_index(drop=True)

    raise FileNotFoundError("Could not find UCI SMS Spam Collection (sms_spam.csv or zip).")


def load_super_sms() -> pd.DataFrame:
    path = _find("super_sms_dataset.csv")
    if path is None:
        raise FileNotFoundError("Could not find Super SMS Dataset (super_sms_dataset.csv).")
    frame = _read_csv(path)
    # Previous project schema: SMSes, Labels (0/1)
    rename = {}
    if "SMSes" in frame.columns:
        rename["SMSes"] = "text"
    if "Labels" in frame.columns:
        rename["Labels"] = "binary_label"
    if "Text" in frame.columns:
        rename["Text"] = "text"
    if "Category" in frame.columns:
        rename["Category"] = "binary_label"
    frame = frame.rename(columns=rename)
    if not {"text", "binary_label"}.issubset(frame.columns):
        raise ValueError(f"Unexpected Super SMS columns: {list(frame.columns)}")
    frame["binary_label"] = _normalize_binary(frame["binary_label"])
    frame = frame.dropna(subset=["text", "binary_label"])
    frame["source"] = "super_sms"
    return frame[["text", "binary_label", "source"]].reset_index(drop=True)


def load_combined_csv() -> pd.DataFrame | None:
    path = _find("sms_combined.csv")
    if path is None:
        return None
    frame = _read_csv(path)
    frame = frame.rename(columns={"Category": "binary_label", "Text": "text"})
    frame["binary_label"] = _normalize_binary(frame["binary_label"])
    frame = frame.dropna(subset=["text", "binary_label"])
    # Keep provenance as combined; source breakdown is recovered when rebuilding.
    frame["source"] = "uci_plus_super_sms"
    return frame[["text", "binary_label", "source"]].reset_index(drop=True)


def merge_uci_and_super(uci: pd.DataFrame, super_sms: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match previous-project merge rules: unique texts; UCI label wins on conflicts."""
    uci = uci.copy()
    super_sms = super_sms.copy()
    uci["text_key"] = uci["text"].astype(str)
    super_sms["text_key"] = super_sms["text"].astype(str)

    uci_unique = uci.drop_duplicates(subset=["text_key"], keep="first")
    super_unique = super_sms.drop_duplicates(subset=["text_key"], keep="first")

    overlap = set(uci_unique["text_key"]).intersection(set(super_unique["text_key"]))
    conflicts = 0
    if overlap:
        uci_map = dict(zip(uci_unique["text_key"], uci_unique["binary_label"]))
        super_map = dict(zip(super_unique["text_key"], super_unique["binary_label"]))
        conflicts = sum(1 for key in overlap if uci_map[key] != super_map[key])

    # Prefer UCI rows for overlapping texts.
    super_only = super_unique[~super_unique["text_key"].isin(overlap)].copy()
    combined = pd.concat([uci_unique, super_only], ignore_index=True)
    combined = combined.drop(columns=["text_key"])
    stats = {
        "uci_rows": int(len(uci_unique)),
        "super_sms_rows": int(len(super_unique)),
        "overlap_texts": int(len(overlap)),
        "label_conflicts_kept_uci": int(conflicts),
        "combined_rows": int(len(combined)),
    }
    return combined, stats


def silver_multiclass(text: str, binary_label: str) -> str:
    if binary_label == "ham":
        return "legitimate"
    signals = extract_signals(text)
    if any(org in ("irs", "ssa", "social security") for org in signals.impersonated_orgs):
        return "government_impersonation"
    if any(org in ("chase", "bank of america", "wells fargo", "paypal") for org in signals.impersonated_orgs):
        return "banking_impersonation"
    joined_money = " ".join(signals.money_mentions).lower()
    if "percent" in joined_money or "%" in joined_money or "earn" in text.lower():
        return "investment_scam"
    if signals.asks_for_sensitive_info:
        return "identity_theft_attempt"
    if signals.urls or signals.urgency_cues:
        return "phishing_spoofing"
    return "other_uncertain"


def main() -> None:
    settings = get_settings()
    store = get_store()
    raw_key = settings.data_raw_key.rstrip("/")
    out_key = settings.processed_key.rstrip("/")
    work_raw = store.work_dir(raw_key)
    work_out = store.work_dir(out_key)

    # Seed raw CSV copies into S3 when present in the code checkout (archive/).
    for name in ("sms_spam.csv", "super_sms_dataset.csv", "sms_combined.csv"):
        local = _find(name)
        if local is not None:
            uri = store.upload_file(local, f"{raw_key}/{name}")
            print("uploaded raw", uri)

    uci = load_uci()
    super_sms = load_super_sms()
    rebuilt, merge_stats = merge_uci_and_super(uci, super_sms)

    # Prefer the previous project's combined file when present and size matches closely.
    packaged = load_combined_csv()
    if packaged is not None and abs(len(packaged) - len(rebuilt)) <= 5:
        frame = packaged.drop_duplicates(subset=["text"]).reset_index(drop=True)
        dataset_name = "UCI SMS Spam Collection + Super SMS Dataset (sms_combined.csv)"
        merge_note = "Loaded packaged sms_combined.csv from the previous project setup."
    else:
        frame = rebuilt.drop_duplicates(subset=["text"]).reset_index(drop=True)
        dataset_name = "UCI SMS Spam Collection + Super SMS Dataset (rebuilt merge)"
        merge_note = "Rebuilt merge from sms_spam.csv + super_sms_dataset.csv."

    # Attach per-row source when rebuilt; packaged combined keeps a single source tag.
    if "source" not in frame.columns:
        frame["source"] = "uci_plus_super_sms"

    print(f"Labeling {len(frame)} messages with silver multi-class heuristics…", flush=True)
    frame["label"] = [
        silver_multiclass(text, binary)
        for text, binary in zip(frame["text"].astype(str), frame["binary_label"].astype(str))
    ]
    frame["label_source"] = frame["binary_label"].map(
        {"ham": "binary_ham", "spam": "silver_from_spam_heuristics"}
    )
    frame["synthetic"] = False

    # Stratified split; fall back if a rare silver class blocks stratification.
    try:
        train_df, temp_df = train_test_split(
            frame, test_size=0.2, random_state=SEED, stratify=frame["label"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
        )
        split_mode = "stratified_by_multiclass_label"
    except ValueError:
        train_df, temp_df = train_test_split(
            frame, test_size=0.2, random_state=SEED, stratify=frame["binary_label"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["binary_label"]
        )
        split_mode = "stratified_by_binary_label"

    outputs = {
        "messages_train.csv": train_df,
        "messages_val.csv": val_df,
        "messages_test.csv": test_df,
        "messages_all.csv": frame,
        "messages_binary_all.csv": frame[["text", "binary_label", "source", "label"]],
    }
    for name, df in outputs.items():
        local = work_out / name
        df.to_csv(local, index=False)
        print("wrote", store.upload_file(local, f"{out_key}/{name}"))

    stats = {
        "available": True,
        "source": "Prototype summary seeded for dashboard wiring (replace with FTC/IC3 tables)",
        "period": "placeholder",
        "storage": "s3",
        "limitations": (
            "These are illustrative category labels for application wiring only. "
            "They are not official FTC/IC3 counts. Replace via scripts/ingest_fraud_stats.py."
        ),
        "categories": [
            {"category": "Imposter scams", "reports": 1, "losses_millions": 0},
            {"category": "Online shopping", "reports": 1, "losses_millions": 0},
            {"category": "Investment", "reports": 1, "losses_millions": 0},
            {"category": "Identity theft", "reports": 1, "losses_millions": 0},
        ],
    }
    print("wrote", store.write_json(f"{out_key}/fraud_stats_summary.json", stats))

    binary_counts = frame["binary_label"].value_counts().to_dict()
    source_counts = frame["source"].value_counts().to_dict()
    meta = {
        "dataset_name": dataset_name,
        "storage": "s3",
        "s3_prefix": store.s3_uri_for(out_key + "/"),
        "datasets_used": [
            {
                "name": "UCI SMS Spam Collection",
                "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
                "rows": merge_stats["uci_rows"],
            },
            {
                "name": "Super SMS Dataset",
                "url": "https://github.com/smspamresearch/spstudy",
                "rows": merge_stats["super_sms_rows"],
            },
        ],
        "merge": merge_stats,
        "merge_note": merge_note,
        "rows": int(len(frame)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "split_mode": split_mode,
        "binary_counts": binary_counts,
        "source_counts": source_counts,
        "label_counts": frame["label"].value_counts().to_dict(),
        "comparison_split": "test",
        "note": (
            "Comparison matrices use the held-out test split of the combined UCI + Super SMS "
            "corpus. Multi-class spam subtypes are silver labels from heuristics; document this."
        ),
    }
    print("wrote", store.write_json(f"{out_key}/split_meta.json", meta))
    # Keep a cache copy for local debugging only.
    (work_raw / ".s3_raw_marker").write_text("raw seeded to S3\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
