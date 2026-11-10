#!/usr/bin/env python3
"""Chunk official guidance from S3 and write BM25 metadata + FAISS index back to S3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def main() -> None:
    settings = get_settings()
    store = get_store()
    knowledge_key = settings.knowledge_key.rstrip("/")
    indexes_key = settings.indexes_key.rstrip("/")

    doc_keys = store.list_keys(f"{knowledge_key}/", suffix=".json")
    if not doc_keys:
        raise FileNotFoundError(
            f"No knowledge docs in S3 under {store.s3_uri_for(knowledge_key + '/')}. "
            "Run: python scripts/seed_knowledge_base.py"
        )

    chunks = []
    for key in doc_keys:
        payload = store.read_json(key)
        for i, piece in enumerate(chunk_text(payload.get("text", ""))):
            chunks.append(
                {
                    "chunk_id": f"{payload.get('doc_id', Path(key).stem)}_{i}",
                    "doc_id": payload.get("doc_id", Path(key).stem),
                    "agency": payload.get("agency"),
                    "title": payload.get("title"),
                    "source_url": payload.get("source_url"),
                    "topic": payload.get("topic"),
                    "retrieved_date": payload.get("retrieved_date"),
                    "text": piece,
                }
            )

    work = store.work_dir(indexes_key)
    chunks_path = work / "chunks.jsonl"
    chunks_path.write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n", encoding="utf-8")
    print(f"Built {len(chunks)} chunks from {len(doc_keys)} S3 docs")

    from sentence_transformers import SentenceTransformer
    import faiss

    model_name = settings.embedding_model
    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss_path = work / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    meta = {
        "embedding_model": model_name,
        "chunks": len(chunks),
        "documents": len(doc_keys),
        "dim": int(vectors.shape[1]),
        "open_source_only": True,
        "storage": "s3",
        "s3_prefix": store.s3_uri_for(indexes_key + "/"),
    }
    meta_path = work / "index_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    uploaded = [
        store.upload_file(chunks_path, f"{indexes_key}/chunks.jsonl"),
        store.upload_file(faiss_path, f"{indexes_key}/faiss.index"),
        store.upload_file(meta_path, f"{indexes_key}/index_meta.json"),
    ]
    for uri in uploaded:
        print("wrote", uri)


if __name__ == "__main__":
    main()
