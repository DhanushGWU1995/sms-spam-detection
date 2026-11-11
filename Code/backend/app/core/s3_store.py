"""S3-backed data store — durable data lives in S3 only; EC2 keeps an ephemeral cache.

All S3 I/O in this project goes through boto3 via ``S3DataStore``. Do not shell out
to the AWS CLI for reads, writes, deletes, or listings.
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from backend.app.core.config import ROOT, get_settings


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


class S3DataStore:
    """Read/write project data under the configured S3 prefix (boto3 only).

    Relative keys match the repo layout (e.g. ``data/knowledge/foo.json``,
    ``artifacts/indexes/faiss.index``). Local files under ``cache_dir`` are a
    disposable working copy for FAISS / joblib / Hugging Face — never the source
    of truth.
    """

    def __init__(self, s3_uri: str | None = None, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        self.s3_uri = (s3_uri or settings.s3_uri).rstrip("/") + "/"
        self.bucket, self.prefix = split_s3_uri(self.s3_uri)
        default_cache = Path(os.environ.get("SCAMSHIELD_CACHE_DIR", tempfile.gettempdir())) / "scamshield-s3-cache"
        self.cache_dir = Path(cache_dir or getattr(settings, "cache_dir", default_cache))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def client(self):
        """Lazy boto3 S3 client (env credentials / instance role / shared config)."""
        if self._client is None:
            region = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-1"
            )
            session = boto3.session.Session()
            self._client = session.client("s3", region_name=region)
        return self._client

    def s3_key(self, relative: str) -> str:
        rel = relative.lstrip("/")
        return f"{self.prefix}{rel}"

    def s3_uri_for(self, relative: str) -> str:
        return f"s3://{self.bucket}/{self.s3_key(relative)}"

    def cache_path(self, relative: str) -> Path:
        return self.cache_dir / relative.lstrip("/")

    def exists(self, relative: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.s3_key(relative))
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def list_keys(self, relative_prefix: str = "", suffix: str | None = None) -> list[str]:
        prefix = self.s3_key(relative_prefix.lstrip("/"))
        keys: list[str] = []
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self.client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []) or []:
                key = item["Key"]
                if key.endswith("/"):
                    continue
                if not key.startswith(self.prefix):
                    continue
                rel = key[len(self.prefix) :]
                if suffix and not rel.endswith(suffix):
                    continue
                keys.append(rel)
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return sorted(keys)

    def list_objects(self, relative_prefix: str = "") -> list[dict[str, Any]]:
        """Return relative key + size + last_modified for objects under a prefix."""
        prefix = self.s3_key(relative_prefix.lstrip("/"))
        rows: list[dict[str, Any]] = []
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self.client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []) or []:
                key = item["Key"]
                if not key.startswith(self.prefix):
                    continue
                rel = key[len(self.prefix) :]
                rows.append(
                    {
                        "key": rel,
                        "size": int(item.get("Size", 0)),
                        "last_modified": item.get("LastModified"),
                    }
                )
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return rows

    def read_bytes(self, relative: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self.s3_key(relative))
        return response["Body"].read()

    def read_text(self, relative: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(relative).decode(encoding)

    def read_json(self, relative: str) -> Any:
        return json.loads(self.read_text(relative))

    def write_bytes(self, relative: str, data: bytes, content_type: str | None = None) -> str:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": self.s3_key(relative), "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)
        dest = self.cache_path(relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return self.s3_uri_for(relative)

    def write_text(self, relative: str, text: str, encoding: str = "utf-8") -> str:
        return self.write_bytes(relative, text.encode(encoding), content_type="text/plain; charset=utf-8")

    def write_json(self, relative: str, payload: Any) -> str:
        body = json.dumps(payload, indent=2) + "\n"
        return self.write_bytes(relative, body.encode("utf-8"), content_type="application/json")

    def ensure_prefix(self, relative_prefix: str) -> str:
        """Create an empty prefix marker object (folder placeholder)."""
        rel = relative_prefix.lstrip("/")
        if not rel.endswith("/"):
            rel += "/"
        self.client.put_object(Bucket=self.bucket, Key=self.s3_key(rel), Body=b"")
        return self.s3_uri_for(rel)

    def upload_file(self, local_path: Path | str, relative: str) -> str:
        local = Path(local_path)
        self.client.upload_file(str(local), self.bucket, self.s3_key(relative))
        dest = self.cache_path(relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local.resolve() != dest.resolve():
            dest.write_bytes(local.read_bytes())
        return self.s3_uri_for(relative)

    def upload_tree(self, local_dir: Path | str, relative_prefix: str) -> list[str]:
        base = Path(local_dir)
        uploaded: list[str] = []
        if not base.exists():
            return uploaded
        prefix = relative_prefix.strip("/")
        for path in base.rglob("*"):
            if path.is_dir():
                continue
            rel = f"{prefix}/{path.relative_to(base).as_posix()}" if prefix else path.relative_to(base).as_posix()
            uploaded.append(self.upload_file(path, rel))
        return uploaded

    def download_file(self, relative: str, force: bool = False) -> Path:
        dest = self.cache_path(relative)
        if dest.exists() and not force:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self.s3_key(relative), str(dest))
        return dest

    def download_prefix(self, relative_prefix: str, force: bool = False) -> Path:
        """Download every object under a prefix into the cache; return the local folder."""
        keys = self.list_keys(relative_prefix)
        if not keys:
            raise FileNotFoundError(f"No S3 objects under {self.s3_uri_for(relative_prefix)}")
        local_root = self.cache_path(relative_prefix.rstrip("/"))
        for rel in keys:
            self.download_file(rel, force=force)
        return local_root

    def delete(self, relative: str) -> str:
        """Delete one object (or empty prefix marker)."""
        key = self.s3_key(relative)
        self.client.delete_object(Bucket=self.bucket, Key=key)
        cached = self.cache_path(relative)
        if cached.exists() and cached.is_file():
            cached.unlink()
        return self.s3_uri_for(relative)

    def delete_keys(self, relatives: Iterable[str]) -> list[str]:
        """Batch-delete relative keys with boto3 ``delete_objects`` (chunks of 1000)."""
        deleted: list[str] = []
        batch: list[dict[str, str]] = []

        def flush() -> None:
            nonlocal batch
            if not batch:
                return
            response = self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": batch, "Quiet": True},
            )
            errors = response.get("Errors") or []
            if errors:
                raise RuntimeError(f"S3 delete_objects errors: {errors[:5]}")
            batch = []

        for rel in relatives:
            rel = rel.lstrip("/")
            batch.append({"Key": self.s3_key(rel)})
            deleted.append(rel)
            cached = self.cache_path(rel)
            if cached.exists() and cached.is_file():
                cached.unlink()
            if len(batch) >= 1000:
                flush()
        flush()
        return deleted

    def delete_prefix(self, relative_prefix: str = "") -> list[str]:
        """Delete every object under a relative prefix (boto3 only)."""
        rows = self.list_objects(relative_prefix)
        keys = [row["key"] for row in rows]
        # Also remove empty folder markers that list_keys skips when ending with "/".
        # list_objects keeps them; delete_keys handles both.
        return self.delete_keys(keys)

    def ensure_local(self, relative: str, force: bool = False) -> Path:
        if not self.exists(relative):
            raise FileNotFoundError(f"Missing S3 object: {self.s3_uri_for(relative)}")
        return self.download_file(relative, force=force)

    def ensure_prefix_local(self, relative_prefix: str, force: bool = False) -> Path:
        return self.download_prefix(relative_prefix, force=force)

    def open_text(self, relative: str):
        return self.ensure_local(relative).open(encoding="utf-8")

    def work_dir(self, relative_prefix: str) -> Path:
        """Ephemeral local directory for script outputs before upload."""
        path = self.cache_path(relative_prefix.rstrip("/"))
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_store() -> S3DataStore:
    return S3DataStore()


def repo_fallback_path(relative: str) -> Path:
    """Legacy on-disk path under the git checkout (code-only; not durable data)."""
    return ROOT / relative.lstrip("/")
