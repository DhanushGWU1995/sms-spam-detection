"""Read-only aggregate fraud statistics helpers (S3-backed)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store


def _stats_key() -> str:
    return f"{get_settings().processed_key.rstrip('/')}/fraud_stats_summary.json"


def load_summary() -> dict[str, Any]:
    store = get_store()
    key = _stats_key()
    if not store.exists(key):
        return {
            "available": False,
            "note": "Aggregate statistics have not been ingested yet.",
            "categories": [],
            "storage": "s3",
        }
    payload = store.read_json(key)
    payload["storage"] = "s3"
    return payload


def query_category(keyword: str = "") -> dict[str, Any]:
    summary = load_summary()
    if not summary.get("available"):
        return summary
    keyword = (keyword or "").strip().lower()
    rows = summary.get("categories", [])
    if keyword:
        rows = [row for row in rows if keyword in str(row.get("category", "")).lower()]
    return {
        "available": True,
        "storage": "s3",
        "source": summary.get("source"),
        "period": summary.get("period"),
        "limitations": summary.get("limitations"),
        "matches": rows[:20],
    }


def dataframe_from_summary() -> pd.DataFrame:
    summary = load_summary()
    rows = summary.get("categories", [])
    return pd.DataFrame(rows)
