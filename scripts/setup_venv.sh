#!/usr/bin/env bash
# Use the shared dl-venv interpreter/torch. Install only missing packages into
# this project's .vendor folder so we do not change dl-venv or reinstall torch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DL_PY="${DL_PY:-/home/ubuntu/dl-venv/bin/python}"
VENDOR="${VENDOR:-$ROOT/.vendor}"

if [[ ! -x "$DL_PY" ]]; then
  echo "dl-venv python not found at $DL_PY"
  exit 1
fi

echo "Using shared env: $DL_PY"
"$DL_PY" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print(sys.executable)
PY

mkdir -p "$VENDOR"
export PYTHONPATH="$ROOT:$VENDOR${PYTHONPATH:+:$PYTHONPATH}"

need=()
pkgs=()
"$DL_PY" -c "import fastapi" >/dev/null 2>&1 || pkgs+=(fastapi "uvicorn[standard]" python-multipart)
"$DL_PY" -c "import pydantic_settings" >/dev/null 2>&1 || pkgs+=(pydantic-settings)
"$DL_PY" -c "import yaml" >/dev/null 2>&1 || pkgs+=(pyyaml)
"$DL_PY" -c "import sentence_transformers" >/dev/null 2>&1 || pkgs+=(sentence-transformers)
"$DL_PY" -c "import faiss" >/dev/null 2>&1 || pkgs+=(faiss-cpu)
"$DL_PY" -c "import rank_bm25" >/dev/null 2>&1 || pkgs+=(rank-bm25)
"$DL_PY" -c "import boto3" >/dev/null 2>&1 || pkgs+=(boto3)
"$DL_PY" -c "import bs4" >/dev/null 2>&1 || pkgs+=(beautifulsoup4 lxml)
"$DL_PY" -c "import httpx" >/dev/null 2>&1 || pkgs+=(httpx)
"$DL_PY" -c "import spacy" >/dev/null 2>&1 || pkgs+=(spacy)

if ((${#pkgs[@]})); then
  echo "Installing into $VENDOR: ${pkgs[*]}"
  "$DL_PY" -m pip install --upgrade --target "$VENDOR" "${pkgs[@]}"
else
  echo "No extra packages needed."
fi

"$DL_PY" - <<PY
import sys
sys.path[:0] = ["$ROOT", "$VENDOR"]
import torch, fastapi, sentence_transformers, faiss, rank_bm25, boto3
print("ready")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("vendor", "$VENDOR")
PY
