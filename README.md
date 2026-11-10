# ScamShield US

Open-source agentic RAG assistant for consumer fraud / identity-theft education.

**Stack constraint:** AWS EC2 (`g5.xlarge`) + Amazon S3 only. No Bedrock, Lambda, API Gateway, RDS, OpenSearch, or SageMaker.

| Layer | Choice |
|---|---|
| Traditional classifiers | TF-IDF + Logistic Regression / Linear SVM |
| Advanced classifier | DistilBERT |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | Hybrid BM25 + FAISS (`top_k=5`) |
| Answer LLM | Qwen2.5-1.5B-Instruct (local) |
| Agents | Deterministic supervisor pipeline in FastAPI |
| API / UI | FastAPI + Angular TypeScript |
| Storage | S3 for all data/models/indexes (EC2 caches ephemerally) |

## Layout

```
backend/app/   FastAPI, NLP, RAG, investigation workflow
frontend/      Angular UI
scripts/       seed / train / index / evaluate / S3 helpers
configs/       settings.yaml (includes s3_uri)
data/          .gitkeep placeholders only
artifacts/     .gitkeep placeholders only
```

## S3 prefix

```
s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/
  data/{raw,processed,knowledge,provenance}/
  artifacts/{models,indexes,metrics,viz}/
```

EC2 keeps code only. Pipeline scripts write outputs to S3; the API reads from S3 into `/tmp/scamshield-s3-cache` (override with `SCAMSHIELD_CACHE_DIR`).

## EC2 quick start

```bash
cd ~/dl-venv/nlp_project_final
bash scripts/setup_venv.sh
bash scripts/run_pipeline.sh

PYTHONPATH="$PWD:$PWD/.vendor" /home/ubuntu/dl-venv/bin/python \
  -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

cd frontend && npm install && npm start
# http://127.0.0.1:4200  (proxies /api → :8000)
```

## Safety

Educational decision support only. Not a legal or law-enforcement determination. The system does not contact senders, submit reports, or access financial accounts.
