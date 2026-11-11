# ScamShield US — System Architecture

## 1. Purpose

ScamShield US is an open-source **agentic RAG** assistant for consumer fraud and identity-theft education. A user pastes a suspicious SMS/email-like message; the system classifies risk, retrieves official guidance, and returns an educational answer with citations.

**Hard stack constraint:** AWS EC2 + Amazon S3 only (no Bedrock, Lambda, API Gateway, RDS, OpenSearch, or SageMaker). All durable data is accessed with **boto3**.

## 2. Logical architecture

```
┌─────────────┐     /api/*      ┌──────────────────────┐
│ Angular UI  │ ──────────────► │ FastAPI (uvicorn)    │
│ :4200       │ ◄────────────── │ Investigation API    │
└─────────────┘   JSON + viz    └──────────┬───────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
              ScamClassifier        HybridRetriever         Qwen LLM
           (sklearn + DistilBERT)   (BM25 + FAISS top_k=5)  (local HF)
                     │                     │                     │
                     └─────────────────────┴─────────────────────┘
                                           │
                                           ▼
                              S3DataStore (boto3)
                     s3://…/nlp-data/{data,artifacts}/
                     ephemeral cache: /tmp/scamshield-s3-cache
```

## 3. Offline pipeline (builds S3 artifacts)

Ordered by `scripts/run_pipeline.sh`:

| Step | Script | S3 outputs |
|---|---|---|
| Init | `sync_s3.py --init` | prefix markers |
| Seed KB | `seed_knowledge_base.py` | `data/knowledge/`, `data/provenance/` |
| Prepare | `prepare_baseline_data.py` | `data/processed/` (+ reads `data/raw/`) |
| Train | `train_classifiers.py` | `artifacts/models/`, `classification_metrics.json` |
| Index | `build_rag_index.py` | `artifacts/indexes/` |
| Evaluate | `evaluate_approaches.py` | `approach_*.json/csv` |
| Viz | `build_dashboard_viz.py` | `artifacts/viz/` |

Environment bootstrap for a new EC2: `bash scripts/setup_venv.sh` → **`uv sync`** from `pyproject.toml` / `uv.lock`.

## 4. Online investigation workflow

`backend.app.agents.workflow.investigate`:

1. **Fraud detection** — Traditional / Advanced / Hybrid classifiers score the message.
2. **Data intelligence** — optional category stats for context.
3. **Retrieval** — hybrid BM25 + dense FAISS; final **top_k = 5** chunks.
4. **Answer generation** — Qwen2.5-1.5B-Instruct answers using only retrieved evidence.
5. **Validation** — soft checks (citations, over-certainty, empty retrieval).

## 5. Classification approaches

- **Traditional NLP:** TF-IDF (+ linguistic features) → Logistic Regression / Linear SVM vote  
- **Advanced NLP:** DistilBERT sequence classifier  
- **Hybrid NLP:** keep Traditional when confident; escalate weak cases to DistilBERT (margin tuned on validation; production eval chose **0.05**)

## 6. RAG design

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Hybrid fusion of BM25 + FAISS dense scores
- Knowledge: ≥20 URL-backed official consumer-protection documents seeded into S3
- Answers are educational and citation-oriented; not legal determinations

## 7. Persistence model

- **GitHub:** code only (backend, frontend, scripts, configs, docs)
- **S3:** raw SMS CSVs, processed splits, knowledge JSON, models, indexes, metrics, viz
- **EC2:** `.venv` via uv; ephemeral S3 cache; no durable project data on disk
