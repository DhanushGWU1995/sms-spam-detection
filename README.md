# ScamShield US — Final Project (Group)

Open-source agentic RAG assistant for consumer fraud / identity-theft education.

**Stack constraint:** AWS EC2 + Amazon S3 only (boto3). Open-source models only — no Bedrock / paid LLM APIs.

| Layer | Choice |
|---|---|
| Traditional classifiers | TF-IDF + Logistic Regression / Linear SVM |
| Advanced classifier | DistilBERT |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | Hybrid BM25 + FAISS (`top_k=5`) |
| Answer LLM | Qwen2.5-1.5B-Instruct (local on EC2) |
| API / UI | FastAPI + Angular TypeScript |
| Storage | S3 via **boto3 only** |
| Python env | **uv sync** from `Code/pyproject.toml` + `Code/uv.lock` |

---

## Repository structure (course deliverables)

This repository contains the four required group folders plus this root README:

```
.
├── README.md                          ← you are here (repo map + how it works)
├── Group-Proposal/
│   └── ScamShield_Group_Proposal.pdf
├── Final-Group-Project-Report/
│   └── ScamShield_Final_Group_Project_Report.pdf
├── Final-Group-Presentation/
│   ├── ScamShield_Final_Group_Presentation.pdf
│   └── ScamShield_Architecture.pptx   ← editable architecture deck
└── Code/
    ├── README.md                      ← run order + script index
    ├── backend/                       ← FastAPI + NLP + RAG + agents
    ├── frontend/                      ← Angular UI
    ├── scripts/                       ← seed → prepare → train → index → eval → viz
    ├── configs/settings.yaml
    ├── pyproject.toml / uv.lock
    ├── data/                          ← placeholders (real data in S3)
    └── artifacts/                     ← placeholders + committed eval metrics / viz
```

Supporting architecture / tech-stack markdown also lives under `docs/` for development notes.

| Folder | Contents |
|---|---|
| **Group-Proposal** | Project proposal PDF (problem, data, methods, packages, tasks, metrics, schedule) |
| **Final-Group-Project-Report** | Full report PDF (sections 1–8 per course rubric) |
| **Final-Group-Presentation** | Presentation PDF + architecture PowerPoint |
| **Code** | All runnable source; see [`Code/README.md`](Code/README.md) for exact run order |

Individual reports and group-contribution writeups are out of scope for this repo layout pass.

---

## How to run the code

All commands run from **`Code/`**. Full details: [`Code/README.md`](Code/README.md).

```bash
cd Code

# 1) Environment
bash scripts/setup_venv.sh
source .venv/bin/activate
export PYTHONPATH="$PWD"

# 2) AWS credentials (course SSO / STS)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-east-1"

# 3) Offline pipeline (only if S3 needs rebuild)
# bash scripts/run_pipeline.sh

# 4) API
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# 5) UI (other terminal)
cd frontend && npm install && npm start
# http://127.0.0.1:4200
```

Pipeline order inside `scripts/run_pipeline.sh`:

1. `sync_s3.py --init` — S3 prefixes  
2. `seed_knowledge_base.py` — official knowledge docs  
3. `prepare_baseline_data.py` — UCI + Super SMS splits  
4. `train_classifiers.py` — Traditional + DistilBERT  
5. `build_rag_index.py` — BM25 + FAISS  
6. `evaluate_approaches.py` — Traditional / Advanced / Hybrid metrics  
7. `build_dashboard_viz.py` — wordclouds / dashboard JSON  

---

## S3 layout (durable store)

```
s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/
  data/{raw,processed,knowledge,provenance}/
  artifacts/{models,indexes,metrics,viz}/
```

EC2 caches under `Code/data/` and `Code/artifacts/` are ephemeral; S3 is the source of truth via boto3.

---

## Held-out test summary

| Approach | Macro-F1 | Accuracy | Routed to Advanced |
|---|---:|---:|---:|
| Traditional NLP | 0.969 | 0.987 | 0% |
| Advanced NLP | 0.961 | 0.987 | 100% |
| **Hybrid NLP** | **0.983** | **0.992** | **1.1%** |

Source: `Code/artifacts/metrics/approach_metrics.csv`.

Educational use only — not legal advice.
