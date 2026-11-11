# ScamShield US — Project Report

## Abstract

ScamShield US is an educational agentic RAG system that classifies suspicious consumer messages and answers user questions using official public guidance. The system runs on AWS EC2 with durable artifacts in S3, using only open-source models and boto3 for storage.

## Problem statement

Consumers receive phishing, government/banking impersonation, investment, and identity-theft themed SMS/messages. They need **decision support** that (1) estimates scam risk and (2) explains recommended actions using trustworthy sources — without paid LLM APIs or managed AWS AI services outside EC2+S3.

## Objectives

1. Compare **Traditional**, **Advanced**, and **Hybrid** NLP classifiers on a shared SMS corpus.
2. Build a hybrid BM25 + FAISS knowledge index over official guidance documents.
3. Serve an Angular + FastAPI product that classifies, retrieves top-5 evidence, and answers with a local Qwen model.
4. Keep **code on GitHub** and **data/models on S3** so a new EC2 can clone + `uv sync` + run.

## Data

- **UCI SMS Spam Collection** and **Super SMS Dataset** uploaded to `data/raw/`.
- Prepared train/val/test splits in `data/processed/` (binary labels from sources; multi-class spam subtypes are silver/heuristic labels where used).
- Knowledge base: 24 URL-backed official documents under `data/knowledge/` with provenance inventory.

## Methods

### Classification
- Traditional: domain TF-IDF + linguistic signals → LR / Linear SVM.
- Advanced: DistilBERT fine-tuned on the project labels.
- Hybrid: cascade Traditional → DistilBERT when Traditional confidence margin is weak (validation-chosen margin **0.05**).
- Interpretability: **LIME** explanations after binary spam/ham prediction (see Results / dashboard).

### Retrieval & generation
- MiniLM embeddings + FAISS dense index fused with BM25.
- Retriever returns **top_k = 5** chunks to the answer LLM.
- Qwen2.5-1.5B-Instruct generates educational answers grounded in those chunks.

### Evaluation
`evaluate_approaches.py` loads trained models and scores the held-out **test** split.

| Approach | Macro-F1 | Accuracy | Weighted-F1 | Routed to Advanced |
|---|---:|---:|---:|---:|
| Traditional NLP | 0.969 | 0.987 | 0.987 | 0% |
| Advanced NLP | 0.961 | 0.987 | 0.987 | 100% |
| **Hybrid NLP** | **0.983** | **0.992** | **0.992** | **1.1%** |

Primary metric is **macro-F1** (rare scam subtypes matter). Accuracy ≈ micro-F1 for single-label classification.

Training also writes `classification_metrics.json`; the UI comparison panel uses `approach_reports.json`.

### Word clouds and LIME (post-prediction)

Dashboard visualizations are produced by `scripts/build_dashboard_viz.py` and shown in this order:

1. **General corpus word clouds** — document-frequency terms in labeled spam vs ham training texts (no model explanation yet).
2. **LIME word clouds after model prediction** — for each sampled message, a binary TF-IDF + logistic regression model predicts spam/ham first; LIME then estimates which tokens pushed that prediction. Aggregated local weights form the LIME spam/ham word clouds (plus example local bar charts).

So LIME tokens are **derived after prediction**, not from raw frequency alone. The Angular UI places corpus clouds first and LIME clouds immediately afterward.

## System delivery

- Backend API + Angular UI
- S3-backed models, indexes, metrics, viz
- README runbook + `scripts/setup_venv.sh` (`uv sync`)
- Architecture deck: `Final-Group-Presentation/ScamShield_Architecture.pptx`

## Limitations & ethics

- Educational decision support only — not legal advice or law-enforcement determination.
- Multi-class spam subtypes beyond binary ham/spam may be silver labels.
- Small open LLM; answers depend on retrieval quality and knowledge coverage.
- Course AWS credentials/session tokens are required and are not stored in GitHub.

## Reproducibility

```bash
git clone https://github.com/DhanushGWU1995/sms-spam-detection.git
cd sms-spam-detection/Code
# export AWS_* credentials
bash scripts/setup_venv.sh
source .venv/bin/activate && export PYTHONPATH="$PWD"
# optional rebuild: bash scripts/run_pipeline.sh
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
