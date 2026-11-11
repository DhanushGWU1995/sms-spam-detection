from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "settings.yaml"


def load_yaml() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class Settings(BaseSettings):
    project_name: str = "ScamShield US"
    s3_uri: str = "s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/"
    seed: int = 42
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_model: str = "distilbert-base-uncased"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Relative keys under s3_uri (durable). Local Path properties are cache mirrors only.
    data_raw_key: str = "data/raw"
    processed_key: str = "data/processed"
    knowledge_key: str = "data/knowledge"
    provenance_key: str = "data/provenance"
    models_key: str = "artifacts/models"
    indexes_key: str = "artifacts/indexes"
    metrics_key: str = "artifacts/metrics"
    viz_key: str = "artifacts/viz"
    sqlite_key: str = "artifacts/scamshield.db"
    cache_dir: Path = Path(tempfile.gettempdir()) / "scamshield-s3-cache"

    class Config:
        env_prefix = "SCAMSHIELD_"

    @property
    def knowledge_dir(self) -> Path:
        return self.cache_dir / self.knowledge_key

    @property
    def index_dir(self) -> Path:
        return self.cache_dir / self.indexes_key

    @property
    def model_dir(self) -> Path:
        return self.cache_dir / self.models_key

    @property
    def metrics_dir(self) -> Path:
        return self.cache_dir / self.metrics_key

    @property
    def processed_dir(self) -> Path:
        return self.cache_dir / self.processed_key

    @property
    def sqlite_path(self) -> Path:
        return self.cache_dir / self.sqlite_key


@lru_cache
def get_settings() -> Settings:
    raw = load_yaml()
    models = raw.get("models", {})
    paths = raw.get("paths", {})
    cache_env = os.environ.get("SCAMSHIELD_CACHE_DIR")
    cache_dir = Path(cache_env) if cache_env else Path(tempfile.gettempdir()) / "scamshield-s3-cache"
    return Settings(
        project_name=raw.get("project_name", "ScamShield US"),
        s3_uri=raw.get("s3_uri", Settings.model_fields["s3_uri"].default),
        seed=int(raw.get("seed", 42)),
        embedding_model=models.get("embedding", Settings.model_fields["embedding_model"].default),
        classifier_model=models.get("classifier", Settings.model_fields["classifier_model"].default),
        llm_model=models.get("llm", Settings.model_fields["llm_model"].default),
        reranker_model=models.get("reranker", Settings.model_fields["reranker_model"].default),
        data_raw_key=paths.get("data_raw", "data/raw"),
        processed_key=paths.get("data_processed", "data/processed"),
        knowledge_key=paths.get("knowledge", "data/knowledge"),
        provenance_key=paths.get("provenance", "data/provenance"),
        models_key=paths.get("models", "artifacts/models"),
        indexes_key=paths.get("indexes", "artifacts/indexes"),
        metrics_key=paths.get("metrics", "artifacts/metrics"),
        viz_key=paths.get("viz", "artifacts/viz"),
        sqlite_key=paths.get("sqlite", "artifacts/scamshield.db"),
        cache_dir=cache_dir,
    )


ROOT_DIR = ROOT
