#!/usr/bin/env python3
"""Build dashboard visualizations: spam/ham word clouds, attention terms, US geo overlay.

SMS rows have almost no state geography, so the map uses educational FTC-style
reference complaint intensity by state (clearly labeled as not SMS-derived).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "frontend" / "src" / "assets" / "viz"

STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was",
    "be", "this", "that", "it", "you", "your", "with", "from", "as", "at", "by", "we",
    "our", "i", "me", "my", "not", "no", "if", "so", "do", "have", "has", "will", "can",
    "just", "get", "got", "u", "ur", "im", "its", "amp", "http", "https", "www", "com",
}


def tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z']{1,}", str(text).lower())
        if tok not in STOP and len(tok) > 2
    ]


def top_terms(texts: list[str], n: int = 25) -> list[dict]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(set(tokenize(text)))  # document frequency style
    total = max(len(texts), 1)
    return [
        {"term": term, "docs": int(docs), "share": float(docs / total)}
        for term, docs in counts.most_common(n)
    ]


def try_wordcloud(spam_texts: list[str], ham_texts: list[str], viz_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from wordcloud import WordCloud
    except Exception:
        return out

    viz_dir.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, texts, color in (
        ("spam_wordcloud.png", spam_texts, "Reds"),
        ("ham_wordcloud.png", ham_texts, "Greens"),
    ):
        text = " ".join(tokenize(" ".join(texts[:8000])))
        if not text.strip():
            continue
        cloud = WordCloud(
            width=900,
            height=480,
            background_color="#101814",
            colormap=color,
            max_words=80,
            prefer_horizontal=0.9,
        ).generate(text)
        for folder in (viz_dir, ASSETS_DIR):
            path = folder / name
            cloud.to_file(str(path))
        # Prefer API static mount (works when Angular assets are not rebuilt).
        out[name.replace(".png", "")] = f"/api/viz/files/{name}"
    return out


def ftc_style_state_intensity() -> list[dict]:
    """Educational reference intensities inspired by FTC Consumer Sentinel volume patterns.

    Not computed from the SMS corpus (which lacks geography). Values are relative 0-100.
    """
    # Relative complaint intensity index (illustrative; CA/FL/TX/NY typically high volume).
    raw = {
        "AL": 28, "AK": 12, "AZ": 48, "AR": 24, "CA": 100, "CO": 42, "CT": 34, "DE": 18,
        "FL": 92, "GA": 58, "HI": 16, "ID": 20, "IL": 55, "IN": 36, "IA": 22, "KS": 23,
        "KY": 30, "LA": 33, "ME": 17, "MD": 40, "MA": 44, "MI": 49, "MN": 35, "MS": 25,
        "MO": 37, "MT": 14, "NE": 19, "NV": 46, "NH": 18, "NJ": 50, "NM": 27, "NY": 88,
        "NC": 52, "ND": 11, "OH": 54, "OK": 29, "OR": 38, "PA": 57, "RI": 16, "SC": 39,
        "SD": 12, "TN": 41, "TX": 95, "UT": 31, "VT": 13, "VA": 47, "WA": 45, "WV": 21,
        "WI": 34, "WY": 10, "DC": 26,
    }
    return [
        {"state": state, "intensity": intensity, "bucket": (
            "very_high" if intensity >= 80 else
            "high" if intensity >= 55 else
            "medium" if intensity >= 30 else
            "low"
        )}
        for state, intensity in sorted(raw.items())
    ]


def fallback_cloud_paths(viz_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("spam_wordcloud.png", "ham_wordcloud.png"):
        if (viz_dir / name).exists() or (ASSETS_DIR / name).exists():
            if not (viz_dir / name).exists() and (ASSETS_DIR / name).exists():
                viz_dir.mkdir(parents=True, exist_ok=True)
                (viz_dir / name).write_bytes((ASSETS_DIR / name).read_bytes())
            out[name.replace(".png", "")] = f"/api/viz/files/{name}"
    return out


def load_train_rows(path: Path) -> tuple[list[str], list[str], list[dict], dict[str, int]]:
    spam: list[str] = []
    ham: list[str] = []
    label_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = str(row.get("text") or "")
            binary = str(row.get("binary_label") or "").strip().lower()
            label = str(row.get("label") or "").strip()
            if label:
                label_counts[label] += 1
            if binary == "spam":
                spam.append(text)
            elif binary == "ham":
                ham.append(text)
    label_mix = [{"label": label, "count": count} for label, count in label_counts.most_common()]
    binary_mix = {"spam": len(spam), "ham": len(ham)}
    return spam, ham, label_mix, binary_mix


def main() -> None:
    settings = get_settings()
    store = get_store()
    viz_key = settings.viz_key.rstrip("/")
    processed_key = settings.processed_key.rstrip("/")
    viz_dir = store.work_dir(viz_key)

    spam: list[str] = []
    ham: list[str] = []
    label_mix: list[dict] = []
    binary_mix = {"spam": 0, "ham": 0}
    attention_note = (
        "Top document-frequency terms in spam vs ham training messages "
        "(proxy for tokens that draw model/attention focus)."
    )

    train_rel = f"{processed_key}/messages_train.csv"
    if store.exists(train_rel):
        train_path = store.ensure_local(train_rel)
        spam, ham, label_mix, binary_mix = load_train_rows(train_path)
    else:
        attention_note = (
            "Training CSV not present in S3; attention terms will be empty until "
            "scripts/prepare_baseline_data.py uploads messages_train.csv."
        )

    cloud_paths = try_wordcloud(spam, ham, viz_dir) if spam or ham else {}
    if not cloud_paths:
        cloud_paths = fallback_cloud_paths(viz_dir)

    payload = {
        "available": True,
        "storage": "s3",
        "wordclouds": cloud_paths,
        "attention_terms": {
            "spam": top_terms(spam, n=20) if spam else [],
            "ham": top_terms(ham, n=20) if ham else [],
            "note": attention_note,
        },
        "label_mix": label_mix,
        "binary_mix": binary_mix,
        "geo": {
            "available": True,
            "source": "Educational FTC-style state complaint intensity overlay",
            "limitations": (
                "The SMS training corpus has almost no usable US state fields, so this map is "
                "NOT derived from message labels. It is an educational geographic overlay of "
                "relative consumer-fraud / ID-theft / scam report intensity patterns for "
                "dashboard storytelling."
            ),
            "metric": "relative_complaint_intensity_0_to_100",
            "states": ftc_style_state_intensity(),
        },
    }
    out = viz_dir / "dashboard_viz.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", store.upload_file(out, f"{viz_key}/dashboard_viz.json"))
    for name in ("spam_wordcloud.png", "ham_wordcloud.png"):
        local = viz_dir / name
        if local.exists():
            print("wrote", store.upload_file(local, f"{viz_key}/{name}"))
    print("wordclouds", cloud_paths)


if __name__ == "__main__":
    main()
