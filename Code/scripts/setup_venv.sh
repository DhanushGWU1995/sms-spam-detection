#!/usr/bin/env bash
# Bootstrap the project env with uv sync from pyproject.toml (+ uv.lock).
#
# Fresh EC2:
#   cd ~/nlp_project_final
#   bash scripts/setup_venv.sh
#   source .venv/bin/activate
#   export PYTHONPATH="$PWD"
#
# Manual equivalent:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
#   uv python install 3.11
#   uv sync          # installs deps from pyproject.toml / uv.lock into .venv
#   source .venv/bin/activate
#   python -m spacy download en_core_web_sm
#
# CPU-only (skip CUDA torch index on Linux):
#   SKIP_CUDA_TORCH=1 bash scripts/setup_venv.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "==> Ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH after install. Add \$HOME/.local/bin to PATH and re-run."
  exit 1
fi
echo "uv $(uv --version)"

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "Missing pyproject.toml in $ROOT"
  exit 1
fi

# Optional: force CPU torch by temporarily ignoring the CUDA index source.
# (Default pyproject.toml pins Linux x86_64 torch to the cu124 index.)
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$ROOT/.venv}"

if [[ "${SKIP_CUDA_TORCH:-0}" == "1" ]]; then
  echo "==> SKIP_CUDA_TORCH=1 — syncing with default PyPI torch (CPU)"
  # Override torch source to PyPI for this run via env-less approach:
  # rewrite marker temporarily is heavy; instead unset CUDA preference by using
  # UV_OVERRIDE / installing after. Simplest reliable path: sync then replace torch.
  uv python install "${PYTHON_VERSION}"
  uv sync --python "${PYTHON_VERSION}"
  uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --reinstall torch
else
  echo "==> uv sync (pyproject.toml / uv.lock → .venv)"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    echo "GPU detected — Linux torch will resolve from pytorch-cu124 (see pyproject.toml)."
  else
    echo "No GPU detected — on macOS/CPU hosts uv resolves default torch; on Linux x86_64 cu124 wheels are still selected by marker."
  fi
  uv python install "${PYTHON_VERSION}"
  uv sync --python "${PYTHON_VERSION}"
fi

# Keep requirements.txt export in sync for readers who expect that file.
if [[ -f "$ROOT/pyproject.toml" ]]; then
  echo "==> Refreshing requirements.txt from the locked uv project"
  uv export --no-dev --no-hashes --no-emit-project -o requirements.txt >/dev/null \
    || uv export --no-hashes -o requirements.txt >/dev/null \
    || true
fi

# shellcheck disable=SC1091
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${HOME}/.local/bin:${PATH}"
PY="${UV_PROJECT_ENVIRONMENT}/bin/python"

echo "==> Installing spaCy English model (en_core_web_sm)"
"${PY}" -m spacy download en_core_web_sm

echo "==> Smoke-checking imports"
"${PY}" - <<'PY'
import sys
print("python", sys.executable, sys.version.split()[0])
import torch
import fastapi
import uvicorn
import pandas
import sklearn
import transformers
import sentence_transformers
import faiss
import rank_bm25
import boto3
import spacy
import yaml
import matplotlib
import wordcloud
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
nlp = spacy.load("en_core_web_sm")
print("spacy", spacy.__version__, "model", nlp.meta.get("name"))
print("ready")
PY

cat <<EOF

Setup complete (uv sync).

Activate before pipeline / API:
  source ${UV_PROJECT_ENVIRONMENT}/bin/activate
  export PYTHONPATH="${ROOT}"

Re-sync later after dependency changes:
  uv sync
  source .venv/bin/activate

Run full pipeline:
  bash scripts/run_pipeline.sh

Start API:
  python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
EOF
