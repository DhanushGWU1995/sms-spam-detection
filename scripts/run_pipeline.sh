#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DL_PY="${DL_PY:-/home/ubuntu/dl-venv/bin/python}"
VENDOR="${VENDOR:-$ROOT/.vendor}"

if [[ ! -x "$DL_PY" ]]; then
  echo "Missing dl-venv python at $DL_PY"
  exit 1
fi

export PYTHONPATH="$ROOT:$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
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
if [[ "${SKIP_LLM_EVAL:-0}" == "1" ]]; then
  "$DL_PY" scripts/evaluate_approaches.py --skip-llm
else
  "$DL_PY" scripts/evaluate_approaches.py
fi

echo "[6/6] Build dashboard viz -> S3"
"$DL_PY" scripts/build_dashboard_viz.py

echo "Pipeline complete. Durable outputs are in S3 (see configs/settings.yaml s3_uri)."
