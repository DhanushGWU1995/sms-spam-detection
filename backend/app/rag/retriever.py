"""Hybrid BM25 + dense FAISS retrieval over official guidance (open-source)."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        settings = get_settings()
        self.indexes_key = settings.indexes_key.rstrip("/")
        self.embedding_name = settings.embedding_model
        self.chunks: list[dict[str, Any]] = []
        self.bm25 = None
        self.faiss_index = None
        self.embedder = None
        self._load()

    def _load(self) -> None:
        store = get_store()
        chunks_rel = f"{self.indexes_key}/chunks.jsonl"
        faiss_rel = f"{self.indexes_key}/faiss.index"
        if not store.exists(chunks_rel):
            return
        chunks_path = store.ensure_local(chunks_rel)
        self.chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        try:
            from rank_bm25 import BM25Okapi

            corpus = [_tokenize(chunk["text"]) for chunk in self.chunks]
            self.bm25 = BM25Okapi(corpus)
        except Exception:
            self.bm25 = None
        if store.exists(faiss_rel):
            import faiss

            faiss_path = store.ensure_local(faiss_rel)
            self.faiss_index = faiss.read_index(str(faiss_path))

    def _ensure_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.embedding_name)

    def ready(self) -> bool:
        return bool(self.chunks)

    def search(self, query: str, top_k: int = 6, mode: str = "hybrid") -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=float)

        if mode in {"bm25", "hybrid"} and self.bm25 is not None:
            bm25_scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=float)
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            scores = scores + (0.45 if mode == "hybrid" else 1.0) * bm25_scores

        if mode in {"dense", "hybrid"} and self.faiss_index is not None:
            self._ensure_embedder()
            vector = self.embedder.encode([query], normalize_embeddings=True)
            denseness, indices = self.faiss_index.search(
                np.asarray(vector, dtype=np.float32), min(top_k * 3, len(self.chunks))
            )
            dense_scores = np.zeros(len(self.chunks), dtype=float)
            for score, idx in zip(denseness[0], indices[0]):
                if idx >= 0:
                    dense_scores[idx] = max(dense_scores[idx], float(score))
            if dense_scores.max() > 0:
                dense_scores = dense_scores / dense_scores.max()
            scores = scores + (0.55 if mode == "hybrid" else 1.0) * dense_scores

        ranked = np.argsort(-scores)[:top_k]
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            hit = dict(self.chunks[int(idx)])
            hit["score"] = float(scores[idx])
            hits.append(hit)
        return hits
