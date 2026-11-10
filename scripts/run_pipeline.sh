#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer project uv venv (.venv). Override with DL_PY=... if needed.
if [[ -z "${DL_PY:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    DL_PY="${ROOT}/.venv/bin/python"
  elif [[ -x /home/ubuntu/dl-venv/bin/python ]]; then
    DL_PY=/home/ubuntu/dl-venv/bin/python
  else
    DL_PY="$(command -v python3)"
  fi
fi

if [[ ! -x "$DL_PY" ]]; then
  echo "No Python found. Run: bash scripts/setup_venv.sh"
  exit 1
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Keep .vendor on path for legacy shared-env installs.
if [[ -d "${ROOT}/.vendor" ]]; then
  export PYTHONPATH="${PYTHONPATH}:${ROOT}/.vendor"
fi

echo "Using $DL_PY"
echo "PYTHONPATH=$PYTHONPATH"
"$DL_PY" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), sys.executable)
PY

echo "[0/6] Ensure S3 data prefixes exist"
"$DL_PY" scripts/sync_s3.py --init

echo "[1/6] Seed knowledge base -> S3"
"$DL_PY" scripts/seed_knowledge_base.py

echo "[2/6] Prepare baseline message splits -> S3"
"$DL_PY" scripts/prepare_baseline_data.py

echo "[3/6] Train classifiers (read/write S3)"
"$DL_PY" scripts/train_classifiers.py

echo "[4/6] Build RAG index -> S3"
"$DL_PY" scripts/build_rag_index.py

echo "[5/6] Evaluate approaches (read/write S3)"
"$DL_PY" scripts/evaluate_approaches.py

echo "[6/6] Build dashboard viz -> S3"
"$DL_PY" scripts/build_dashboard_viz.py

echo "Pipeline complete. Durable outputs are in S3 (see configs/settings.yaml s3_uri)."
