#!/usr/bin/env python3
"""Bulk mirror helpers for the course S3 prefix (boto3 only — no AWS CLI).

Durable data always lives in S3. Prefer script-native S3 writes
(seed / build_rag_index / train / evaluate). Use this tool only for
emergency full-tree upload/download, listing, deletes, or empty-prefix bootstrap.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.s3_store import S3DataStore, get_store

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
)


def _store(uri: str | None) -> S3DataStore:
    if uri:
        return S3DataStore(uri)
    return get_store()


def upload_repo_data(uri: str | None = None) -> None:
    """Upload local data/artifacts trees into S3 (never code/docs/configs)."""
    store = _store(uri)
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
    store = _store(uri)
    keys = store.list_keys("")
    for rel in keys:
        if not (rel.startswith("data/") or rel.startswith("artifacts/")):
            continue
        path = store.download_file(rel, force=True)
        print(f"download {store.s3_uri_for(rel)} -> {path}")


def ensure_prefixes(uri: str | None = None) -> None:
    store = _store(uri)
    for name in DATA_PREFIXES:
        print("ensure", store.ensure_prefix(name))


def list_prefix(uri: str | None = None, relative: str = "") -> None:
    store = _store(uri)
    rows = store.list_objects(relative)
    if not rows:
        print(f"(empty) {store.s3_uri_for(relative)}")
        return
    for row in rows:
        ts = row["last_modified"]
        stamp = ts.strftime("%Y-%m-%d %H:%M:%S") if ts is not None else "-"
        print(f"{stamp}  {row['size']:>12}  {store.s3_uri_for(row['key'])}")


def delete_paths(uri: str | None, relatives: list[str], recursive: bool) -> None:
    store = _store(uri)
    for rel in relatives:
        rel = rel.lstrip("/")
        if recursive or rel.endswith("/"):
            deleted = store.delete_prefix(rel)
            for key in deleted:
                print("delete", store.s3_uri_for(key))
            if not deleted:
                # Still try deleting a bare marker key.
                print("delete", store.delete(rel if rel.endswith("/") else rel + "/"))
        else:
            print("delete", store.delete(rel))


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="S3 data mirror utilities for ScamShield (boto3 only)")
    parser.add_argument("--uri", default=os.environ.get("SCAMSHIELD_S3_URI", settings.s3_uri))
    parser.add_argument("--upload", action="store_true", help="Upload local data/artifacts into S3")
    parser.add_argument("--download", action="store_true", help="Download S3 data into ephemeral cache")
    parser.add_argument("--init", action="store_true", help="Create empty S3 data prefixes")
    parser.add_argument("--list", nargs="?", const="", default=None, metavar="PREFIX", help="List objects under PREFIX (boto3)")
    parser.add_argument("--rm", nargs="+", metavar="KEY", help="Delete object key(s) via boto3")
    parser.add_argument("--recursive", action="store_true", help="With --rm, delete entire prefixes")
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
    if args.list is not None:
        list_prefix(args.uri, args.list)
    if args.rm:
        delete_paths(args.uri, args.rm, args.recursive)
    if not any([args.init, args.upload, args.download, args.list is not None, args.rm]):
        parser.print_help()
        print("\nConfigured S3 URI:", settings.s3_uri)
        print("Runtime cache:", settings.cache_dir)
        print("S3 access: boto3 only (backend.app.core.s3_store)")


if __name__ == "__main__":
    main()
