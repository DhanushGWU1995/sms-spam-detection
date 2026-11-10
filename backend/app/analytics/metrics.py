"""Load saved approach-comparison metrics for the API / UI (from S3)."""

from __future__ import annotations

import json
from typing import Any

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store


def _metrics_key(name: str) -> str:
    return f"{get_settings().metrics_key.rstrip('/')}/{name}"


def _viz_key(name: str) -> str:
    return f"{get_settings().viz_key.rstrip('/')}/{name}"


def label_breakdown_from_confusion(confusion: list[list[int]], labels: list[str]) -> list[dict[str, Any]]:
    """One-vs-rest TP/FP/FN/TN + precision/recall/F1 for each label."""
    matrix = confusion or []
    n = len(labels)
    total = int(sum(sum(row) for row in matrix)) if matrix else 0
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        if i >= len(matrix):
            break
        tp = int(matrix[i][i]) if i < len(matrix[i]) else 0
        fp = int(sum(matrix[r][i] for r in range(n) if r < len(matrix) and i < len(matrix[r]) and r != i))
        fn = int(sum(matrix[i][c] for c in range(n) if c < len(matrix[i]) and c != i))
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = tp + fn
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def _enrich_approach_reports(payload: dict[str, Any]) -> dict[str, Any]:
    labels = list(payload.get("labels") or [])
    splits = payload.get("splits") or {}
    for split_name, split in splits.items():
        approaches = (split or {}).get("approaches") or {}
        for name, row in approaches.items():
            confusion = row.get("confusion") or []
            row["label_metrics"] = label_breakdown_from_confusion(confusion, labels)
            report = row.get("report") or {}
            # Prefer sklearn report f1 when present.
            for item in row["label_metrics"]:
                rep = report.get(item["label"]) or {}
                if "f1-score" in rep:
                    item["f1"] = float(rep["f1-score"])
                if "precision" in rep:
                    item["precision"] = float(rep["precision"])
                if "recall" in rep:
                    item["recall"] = float(rep["recall"])
                if "support" in rep:
                    item["support"] = int(rep["support"])
    return payload


def load_classification_metrics() -> dict[str, Any]:
    store = get_store()
    key = _metrics_key("classification_metrics.json")
    if not store.exists(key):
        return {"available": False, "note": "Train classifiers first."}
    payload = store.read_json(key)
    payload["available"] = True
    payload["storage"] = "s3"
    return payload


def load_approach_reports() -> dict[str, Any]:
    store = get_store()
    key = _metrics_key("approach_reports.json")
    if not store.exists(key):
        return {
            "available": False,
            "note": (
                "Approach comparison has not been generated yet. "
                "Run: python scripts/evaluate_approaches.py"
            ),
        }
    payload = store.read_json(key)
    payload = _enrich_approach_reports(payload)
    payload["available"] = True
    payload["storage"] = "s3"
    return payload


def load_dashboard_viz() -> dict[str, Any]:
    store = get_store()
    key = _viz_key("dashboard_viz.json")
    if not store.exists(key):
        return {
            "available": False,
            "note": "Run: python scripts/build_dashboard_viz.py",
        }
    payload = store.read_json(key)
    payload["available"] = True
    payload["storage"] = "s3"
    return payload


def load_knowledge_sources() -> dict[str, Any]:
    """Official sites / pages used to seed the RAG knowledge base (S3)."""
    settings = get_settings()
    store = get_store()
    knowledge_key = settings.knowledge_key.rstrip("/")
    indexes_key = settings.indexes_key.rstrip("/")
    provenance_key = f"{settings.provenance_key.rstrip('/')}/knowledge_inventory.json"
    documents: list[dict[str, Any]] = []

    if store.exists(provenance_key):
        payload = store.read_json(provenance_key)
        documents = list(payload.get("documents") or [])
        retrieved_date = payload.get("retrieved_date")
    else:
        retrieved_date = None
        for key in store.list_keys(f"{knowledge_key}/", suffix=".json"):
            doc = store.read_json(key)
            documents.append(
                {
                    "doc_id": doc.get("doc_id") or key.rsplit("/", 1)[-1].removesuffix(".json"),
                    "agency": doc.get("agency"),
                    "title": doc.get("title"),
                    "source_url": doc.get("source_url"),
                    "topic": doc.get("topic"),
                    "retrieved_date": doc.get("retrieved_date"),
                    "s3_uri": store.s3_uri_for(key),
                }
            )
            if not retrieved_date:
                retrieved_date = doc.get("retrieved_date")

    chunks: list[dict[str, Any]] = []
    index_meta: dict[str, Any] = {}
    meta_key = f"{indexes_key}/index_meta.json"
    chunks_key = f"{indexes_key}/chunks.jsonl"
    if store.exists(meta_key):
        index_meta = store.read_json(meta_key)
    if store.exists(chunks_key):
        for line in store.read_text(chunks_key).splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = str(chunk.get("text") or "")
            chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "agency": chunk.get("agency"),
                    "title": chunk.get("title"),
                    "topic": chunk.get("topic"),
                    "source_url": chunk.get("source_url"),
                    "retrieved_date": chunk.get("retrieved_date"),
                    "char_count": len(text),
                    "text": text,
                }
            )

    if not documents and not chunks:
        return {
            "available": False,
            "note": "No RAG knowledge docs found in S3. Run: python scripts/seed_knowledge_base.py",
            "documents": [],
            "chunks": [],
            "storage": "s3",
            "s3_uri": store.s3_uri,
        }

    return {
        "available": True,
        "storage": "s3",
        "s3_uri": store.s3_uri,
        "retrieved_date": retrieved_date,
        "note": (
            "Curated public US government consumer-fraud guidance pages used by the RAG agent "
            "(stored under the course S3 prefix; fetched by EC2 at runtime)."
        ),
        "documents": documents,
        "count": len(documents),
        "index": {
            "available": bool(chunks),
            "chunk_count": len(chunks),
            "embedding_model": index_meta.get("embedding_model"),
            "dim": index_meta.get("dim"),
            "faiss_type": "IndexFlatIP",
            "total_chars": int(sum(int(c.get("char_count") or 0) for c in chunks)),
            "s3_prefix": store.s3_uri_for(indexes_key + "/"),
        },
        "chunks": chunks,
    }


def load_llm_summary() -> dict[str, Any]:
    # Classification no longer uses an LLM cascade; keep endpoint for compatibility.
    store = get_store()
    key = _metrics_key("approach_summary.json")
    if store.exists(key):
        payload = store.read_json(key)
        payload["available"] = True
        payload["note"] = "Classification cascade is Traditional → Advanced only (no LLM stage)."
        return payload
    return {"available": False, "note": "LLM classification cascade removed."}


def weak_confidence_margin(default: float = 0.10) -> float:
    store = get_store()
    reports = load_approach_reports()
    if reports.get("available"):
        margin = reports.get("hybrid_margin")
        if margin is None:
            margin = (reports.get("summary") or {}).get("hybrid_margin")
        if margin is not None:
            return float(margin)
    key = _metrics_key("approach_summary.json")
    if store.exists(key):
        summary = store.read_json(key)
        margin = summary.get("hybrid_margin")
        if margin is not None:
            return float(margin)
    return default
