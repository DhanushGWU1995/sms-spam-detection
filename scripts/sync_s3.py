#!/usr/bin/env python3
"""Bulk mirror helpers for the course S3 prefix.

Durable data always lives in S3. Prefer script-native S3 writes
(seed / build_rag_index / train / evaluate). Use this tool only for
emergency full-tree upload/download or empty-prefix bootstrap.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store

ROOT = Path(__file__).resolve().parents[1]
DATA_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "data/knowledge/",
    "data/provenance/",
    "artifacts/models/",
    "artifacts/indexes/",
    "artifacts/metrics/",
    "artifacts/viz/",
    "artifacts/backups/",
)


def upload_repo_data(uri: str | None = None) -> None:
    """Upload any leftover local data/artifacts trees into S3 (migration aid)."""
    store = get_store() if uri is None else __import__("backend.app.core.s3_store", fromlist=["S3DataStore"]).S3DataStore(uri)
    for folder in ("data", "artifacts"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_dir() or path.name == ".gitkeep":
                continue
            if path.suffix in {".pem", ".env"} or path.name.startswith(".env"):
                continue
            rel = path.relative_to(ROOT).as_posix()
            print("upload", store.upload_file(path, rel))


def download_all(uri: str | None = None) -> None:
    store = get_store() if uri is None else __import__("backend.app.core.s3_store", fromlist=["S3DataStore"]).S3DataStore(uri)
    keys = store.list_keys("")
    for rel in keys:
        # Skip non-data objects accidentally under the prefix.
        if not (rel.startswith("data/") or rel.startswith("artifacts/") or rel.startswith("configs/") or rel.startswith("docs/")):
            continue
        path = store.download_file(rel, force=True)
        print(f"download {store.s3_uri_for(rel)} -> {path}")


def ensure_prefixes(uri: str | None = None) -> None:
    store = get_store() if uri is None else __import__("backend.app.core.s3_store", fromlist=["S3DataStore"]).S3DataStore(uri)
    for name in DATA_PREFIXES:
        key = store.s3_key(name)
        print(f"ensure s3://{store.bucket}/{key}")
        store.client.put_object(Bucket=store.bucket, Key=key, Body=b"")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="S3 data mirror utilities for ScamShield")
    parser.add_argument("--uri", default=os.environ.get("SCAMSHIELD_S3_URI", settings.s3_uri))
    parser.add_argument("--upload", action="store_true", help="Upload local data/artifacts into S3")
    parser.add_argument("--download", action="store_true", help="Download S3 data into ephemeral cache")
    parser.add_argument("--init", action="store_true", help="Create empty S3 data prefixes")
    args = parser.parse_args()
    if args.uri:
        os.environ["SCAMSHIELD_S3_URI"] = args.uri
        get_store.cache_clear()
    if args.init:
        ensure_prefixes(args.uri)
    if args.upload:
        upload_repo_data(args.uri)
    if args.download:
        download_all(args.uri)
    if not any([args.init, args.upload, args.download]):
        parser.print_help()
        print("\nConfigured S3 URI:", settings.s3_uri)
        print("Runtime cache:", settings.cache_dir)


if __name__ == "__main__":
    main()
