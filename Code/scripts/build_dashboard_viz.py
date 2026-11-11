#!/usr/bin/env python3
"""Build dashboard visualizations: spam/ham word clouds, LIME explanations, geo overlay.

SMS rows have almost no state geography, so the map uses educational FTC-style
reference complaint intensity by state (clearly labeled as not SMS-derived).

LIME (Local Interpretable Model-agnostic Explanations) is fit on a binary
spam/ham TF-IDF + logistic regression surrogate so local word contributions can
be shown next to corpus word clouds.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
    for name in (
        "spam_wordcloud.png",
        "ham_wordcloud.png",
        "spam_lime_wordcloud.png",
        "ham_lime_wordcloud.png",
        "spam_lime_example.png",
        "ham_lime_example.png",
    ):
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


def _save_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="#101814")
    plt.close()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / path.name).write_bytes(path.read_bytes())


def _lime_wordcloud_from_weights(
    weights: dict[str, float],
    path: Path,
    colormap: str,
) -> bool:
    try:
        from wordcloud import WordCloud
    except Exception:
        return False
    positive = {tok: float(w) for tok, w in weights.items() if w > 0}
    if not positive:
        return False
    cloud = WordCloud(
        width=900,
        height=480,
        background_color="#101814",
        colormap=colormap,
        max_words=60,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(positive)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_file(str(path))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / path.name).write_bytes(path.read_bytes())
    return True


def build_lime_bundle(
    spam_texts: list[str],
    ham_texts: list[str],
    viz_dir: Path,
    *,
    n_explain: int = 40,
    num_features: int = 12,
) -> dict:
    """Fit binary spam/ham LR and produce LIME example plots + LIME word clouds."""
    empty = {
        "available": False,
        "note": "LIME artifacts unavailable (need both spam and ham training texts).",
        "images": {},
        "top_features": {"spam": [], "ham": []},
        "examples": [],
    }
    if len(spam_texts) < 5 or len(ham_texts) < 5:
        return empty

    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as exc:  # pragma: no cover
        empty["note"] = f"lime package not installed: {exc}"
        return empty

    # Cap fitting corpus for speed on CPU while keeping class balance.
    spam_fit = spam_texts[:4000]
    ham_fit = ham_texts[:4000]
    texts = spam_fit + ham_fit
    y = np.array([1] * len(spam_fit) + [0] * len(ham_fit))

    pipe = make_pipeline(
        TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        ),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    )
    pipe.fit(texts, y)

    explainer = LimeTextExplainer(class_names=["ham", "spam"], random_state=42)
    spam_weights: dict[str, float] = defaultdict(float)
    ham_weights: dict[str, float] = defaultdict(float)
    spam_counts: Counter[str] = Counter()
    ham_counts: Counter[str] = Counter()
    examples: list[dict] = []

    viz_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def _sample(pool: list[str], k: int) -> list[str]:
        if len(pool) <= k:
            return list(pool)
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in idx]

    spam_samples = _sample(spam_texts, n_explain)
    ham_samples = _sample(ham_texts, n_explain)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind, samples, target_label, weight_map, count_map in (
            ("spam", spam_samples, 1, spam_weights, spam_counts),
            ("ham", ham_samples, 0, ham_weights, ham_counts),
        ):
            for i, text in enumerate(samples):
                clipped = " ".join(str(text).split()[:80])
                if len(clipped) < 8:
                    continue
                try:
                    exp = explainer.explain_instance(
                        clipped,
                        pipe.predict_proba,
                        num_features=num_features,
                        labels=(target_label,),
                    )
                except Exception:
                    continue
                pairs = exp.as_list(label=target_label)
                for tok, w in pairs:
                    token = str(tok).strip().lower()
                    if not token or token in STOP or len(token) < 2:
                        continue
                    weight_map[token] += float(w)
                    count_map[token] += 1

                # Persist one representative bar chart per class.
                if i == 0:
                    fig = exp.as_pyplot_figure(label=target_label)
                    fig.patch.set_facecolor("#101814")
                    ax = fig.axes[0]
                    ax.set_facecolor("#101814")
                    ax.tick_params(colors="#cfe0d6")
                    ax.xaxis.label.set_color("#cfe0d6")
                    ax.yaxis.label.set_color("#cfe0d6")
                    for spine in ax.spines.values():
                        spine.set_color("#2a3a32")
                    ax.set_title(
                        f"LIME local explanation ({kind} example)",
                        color="#e8f5ee",
                        fontsize=11,
                    )
                    out_name = f"{kind}_lime_example.png"
                    _save_png(viz_dir / out_name)
                    examples.append(
                        {
                            "class": kind,
                            "text_preview": clipped[:220],
                            "prediction": kind,
                            "features": [
                                {"term": str(t), "weight": float(w)} for t, w in pairs[:10]
                            ],
                            "image": f"/api/viz/files/{out_name}",
                        }
                    )

    def _ranked(weight_map: dict[str, float], count_map: Counter[str], n: int = 15) -> list[dict]:
        ranked = sorted(weight_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
        out = []
        for term, total_w in ranked[:n]:
            c = max(count_map[term], 1)
            out.append(
                {
                    "term": term,
                    "mean_weight": float(total_w / c),
                    "total_weight": float(total_w),
                    "explanations": int(c),
                }
            )
        return out

    images: dict[str, str] = {}
    for kind, weight_map, cmap in (
        ("spam", spam_weights, "Reds"),
        ("ham", ham_weights, "Greens"),
    ):
        # Keep tokens that push toward the target class on average.
        signed = {
            t: (total / max(spam_counts[t] if kind == "spam" else ham_counts[t], 1))
            for t, total in weight_map.items()
        }
        # For spam cloud: positive = pushes toward spam; for ham: positive toward ham
        # (LIME weights for the explained label).
        path = viz_dir / f"{kind}_lime_wordcloud.png"
        if _lime_wordcloud_from_weights(signed, path, cmap):
            images[f"{kind}_lime_wordcloud"] = f"/api/viz/files/{path.name}"
        example_path = viz_dir / f"{kind}_lime_example.png"
        if example_path.exists():
            images[f"{kind}_lime_example"] = f"/api/viz/files/{example_path.name}"

    return {
        "available": bool(images),
        "method": "LIME (LimeTextExplainer) over TF-IDF + LogisticRegression binary spam/ham",
        "note": (
            "Corpus word clouds show document-frequency prominence. LIME word clouds aggregate "
            "local feature weights from LIME explanations on held-in training samples, showing "
            "which tokens push predictions toward spam or ham."
        ),
        "n_spam_explained": len(spam_samples),
        "n_ham_explained": len(ham_samples),
        "num_features": num_features,
        "images": images,
        "top_features": {
            "spam": _ranked(spam_weights, spam_counts),
            "ham": _ranked(ham_weights, ham_counts),
        },
        "examples": examples,
    }


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
        "(corpus frequency view; pair with LIME for local model explanations)."
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

    print("Building LIME explanations alongside word clouds…", flush=True)
    lime_payload = (
        build_lime_bundle(spam, ham, viz_dir) if spam and ham else {
            "available": False,
            "note": "Need spam and ham rows for LIME.",
            "images": {},
            "top_features": {"spam": [], "ham": []},
            "examples": [],
        }
    )
    # Merge any LIME images into wordclouds map for simple UI fallbacks.
    for key, url in (lime_payload.get("images") or {}).items():
        cloud_paths[key] = url

    payload = {
        "available": True,
        "storage": "s3",
        "wordclouds": cloud_paths,
        "lime": lime_payload,
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
    for name in (
        "spam_wordcloud.png",
        "ham_wordcloud.png",
        "spam_lime_wordcloud.png",
        "ham_lime_wordcloud.png",
        "spam_lime_example.png",
        "ham_lime_example.png",
    ):
        local = viz_dir / name
        if local.exists():
            print("wrote", store.upload_file(local, f"{viz_key}/{name}"))
    print("wordclouds", cloud_paths)
    print("lime_available", lime_payload.get("available"))


if __name__ == "__main__":
    main()
