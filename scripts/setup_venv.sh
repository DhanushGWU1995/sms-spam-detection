#!/usr/bin/env bash
# Create a project-local Python env with uv and install everything needed
# to run the full ScamShield pipeline + API on a fresh EC2 (or laptop).
#
# Fresh EC2 (recommended):
#   cd ~/nlp_project_final   # or wherever you cloned the repo
#   bash scripts/setup_venv.sh
#
# Manual equivalent:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
#   uv python install 3.11
#   uv venv --python 3.11 .venv
#   source .venv/bin/activate
#   # GPU torch (g5 / CUDA). Skip on CPU-only machines — setup detects this.
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
#   uv pip install -r requirements.txt
#   python -m spacy download en_core_web_sm
#
# Then always:
#   source .venv/bin/activate
#   export PYTHONPATH="$PWD"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

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

echo "==> Creating venv at ${VENV_DIR} (Python ${PYTHON_VERSION})"
uv python install "${PYTHON_VERSION}" >/dev/null
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PATH="${VENV_DIR}/bin:${HOME}/.local/bin:${PATH}"
PY="${VENV_DIR}/bin/python"

echo "==> Detecting GPU / CUDA"
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi

if [[ "${HAS_GPU}" -eq 1 && "${SKIP_CUDA_TORCH:-0}" != "1" ]]; then
  echo "==> Installing CUDA PyTorch from ${TORCH_CUDA_INDEX}"
  uv pip install --python "${PY}" torch --index-url "${TORCH_CUDA_INDEX}"
else
  echo "==> Installing CPU PyTorch (no GPU detected, or SKIP_CUDA_TORCH=1)"
  uv pip install --python "${PY}" torch
fi

echo "==> Installing project requirements with uv"
uv pip install --python "${PY}" -r requirements.txt

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

Setup complete.

Activate before pipeline / API:
  source ${VENV_DIR}/bin/activate
  export PYTHONPATH="${ROOT}"

Run full pipeline:
  bash scripts/run_pipeline.sh

Start API:
  python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
EOF
