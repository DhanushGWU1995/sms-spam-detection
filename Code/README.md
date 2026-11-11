# Code — ScamShield US

All project source code lives in this folder. Durable data/models are in S3 (boto3); this tree is code + placeholders.

## Run order

Run commands from **this `Code/` directory** (or set `PYTHONPATH` to it).

### 1) Environment (first time on a machine)

```bash
cd Code
bash scripts/setup_venv.sh
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

Uses **`uv sync`** from `pyproject.toml` / `uv.lock`.

### 2) AWS credentials

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-east-1"
```

### 3) Offline pipeline (builds / refreshes S3 artifacts)

```bash
bash scripts/run_pipeline.sh
```

Order inside the pipeline:

| Order | Script | Purpose |
|---:|---|---|
| 0 | `scripts/sync_s3.py --init` | Ensure S3 prefixes |
| 1 | `scripts/seed_knowledge_base.py` | Seed official knowledge docs → S3 |
| 2 | `scripts/prepare_baseline_data.py` | Build train/val/test from UCI + Super SMS → S3 |
| 3 | `scripts/train_classifiers.py` | Train Traditional + DistilBERT → S3 |
| 4 | `scripts/build_rag_index.py` | Build BM25 + FAISS index → S3 |
| 5 | `scripts/evaluate_approaches.py` | Compare approaches on test → S3 metrics |
| 6 | `scripts/build_dashboard_viz.py` | Wordclouds / dashboard JSON → S3 |

If S3 already has models/indexes/metrics, you can skip step 3 and go to the API.

### 4) Backend API

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 5) Frontend UI

```bash
cd frontend
npm install
npm start
# http://127.0.0.1:4200  (proxies /api → :8000)
```

## Script index

| Path | Description |
|---|---|
| `scripts/setup_venv.sh` | Install uv env (`uv sync`) |
| `scripts/run_pipeline.sh` | Full offline pipeline |
| `scripts/sync_s3.py` | List / init / upload / download / delete via boto3 |
| `scripts/seed_knowledge_base.py` | Official knowledge seeding |
| `scripts/prepare_baseline_data.py` | Dataset merge + splits |
| `scripts/train_classifiers.py` | sklearn + DistilBERT training |
| `scripts/build_rag_index.py` | Hybrid retrieval index |
| `scripts/evaluate_approaches.py` | Traditional / Advanced / Hybrid eval |
| `scripts/build_dashboard_viz.py` | Dashboard visualizations |
| `backend/app/` | FastAPI API, classifiers, RAG, agent workflow |
| `frontend/` | Angular TypeScript UI |
| `configs/settings.yaml` | Models, paths, S3 URI |

## Layout

```
Code/
  backend/app/     FastAPI + NLP + RAG + agents
  frontend/        Angular UI
  scripts/         Pipeline & S3 helpers
  configs/         settings.yaml
  data/            .gitkeep placeholders (real data in S3)
  artifacts/       .gitkeep placeholders (real models in S3)
  pyproject.toml   uv dependency source of truth
  uv.lock          locked versions
  requirements.txt exported from uv.lock
```
