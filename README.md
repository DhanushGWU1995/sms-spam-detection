# ScamShield US

An agentic RAG-based consumer fraud and identity-theft intelligence system using open-source NLP and LLMs.

Deployment constraint: **AWS EC2 g5.xlarge + Amazon S3 only**. No Bedrock, Lambda, API Gateway, RDS, OpenSearch, or SageMaker.

## Open-source stack (free / current)

| Layer | Choice |
|---|---|
| Classifier baselines | scikit-learn TF-IDF + Logistic Regression / Linear SVM / Multinomial NB |
| Transformer classifier | DistilBERT (`distilbert-base-uncased`) |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` |
| Lexical retrieval | BM25 (`rank_bm25`) |
| Dense retrieval | FAISS (CPU) |
| Local LLM | Qwen2.5 Instruct (Hugging Face, Apache 2.0) — start with **1.5B**, optional **7B 4-bit** if VRAM allows |
| Agents | LangGraph |
| API | FastAPI + Uvicorn |
| UI | Angular 19 + TypeScript (`frontend/`) |
| Storage | **Amazon S3 only** for data/models/indexes (EC2 keeps ephemeral cache) |

No paid model APIs are required.

## Pipelines

1. **Consumer intelligence** — structured US fraud statistics (FTC / IC3 / BJS) and message classification.
2. **Conversational intelligence** — hybrid RAG over official guidance + local LLM + supervisor agents.

## Repository layout

```
backend/app/          FastAPI service, NLP, RAG, agents
frontend/             Angular 19 TypeScript UI
scripts/              data download, S3 sync, train, build index
notebooks/            EDA and evaluation
data/                 placeholder dirs only (.gitkeep) — durable data is in S3
artifacts/            placeholder dirs only — models/indexes/metrics live in S3
configs/              YAML settings (includes s3_uri)
archive/sms_spam_v1/  previous SMS-spam course prototype (seed input only)
```

## S3 is the data store

EC2 should keep **code only**. All datasets, knowledge docs, FAISS indexes, trained models, and metrics are written to / read from:

```
s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/
  data/raw/
  data/processed/
  data/knowledge/
  data/provenance/
  artifacts/models/
  artifacts/indexes/
  artifacts/metrics/
  artifacts/viz/
```

Pipeline scripts running on EC2 write directly to that prefix. The FastAPI backend fetches objects from S3 into an ephemeral cache (`SCAMSHIELD_CACHE_DIR`, default `/tmp/scamshield-s3-cache`) for FAISS / joblib / DistilBERT, then serves the UI.

```bash
# Seed 20+ knowledge docs into S3, then build FAISS chunks/index into S3
python scripts/seed_knowledge_base.py
python scripts/build_rag_index.py
```

## Quick start (EC2)

```bash
cd ~/dl-venv/nlp_project_final

# Reuse shared dl-venv torch/CUDA. Extra packages go into ./.vendor only.
bash scripts/setup_venv.sh
bash scripts/run_pipeline.sh   # all outputs go to S3

# API (reads models/indexes/knowledge from S3)
PYTHONPATH="$PWD:$PWD/.vendor" /home/ubuntu/dl-venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Angular TypeScript UI (separate terminal)
cd frontend
npm install
npm start
# open http://127.0.0.1:4200  (proxies /api to FastAPI :8000)
```

## Safety

ScamShield is educational decision support. It is not a law-enforcement determination, legal opinion, or guarantee that a message is safe or fraudulent. Agents never submit reports, contact alleged scammers, or access financial accounts.
