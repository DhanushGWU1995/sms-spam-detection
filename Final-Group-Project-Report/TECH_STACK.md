# ScamShield US — Technology Stack

## Infrastructure

| Component | Choice |
|---|---|
| Compute | AWS EC2 (`g5.xlarge` recommended) |
| Object storage | Amazon S3 (`s3://dats-nlp/…/nlp-data/`) |
| S3 SDK | **boto3 only** (`backend.app.core.s3_store.S3DataStore`) |
| Forbidden | Bedrock, Lambda, API Gateway, RDS, OpenSearch, SageMaker |

## Application

| Layer | Technology |
|---|---|
| UI | Angular 19 + TypeScript |
| API | FastAPI + Uvicorn |
| Agent orchestration | Deterministic supervisor pipeline in Python (`investigate`) |
| Config | `configs/settings.yaml` + pydantic-settings |

## Machine learning / NLP

| Role | Model / library |
|---|---|
| Traditional classifiers | scikit-learn (TF-IDF, Logistic Regression, Linear SVM) |
| Advanced classifier | DistilBERT (`distilbert-base-uncased` fine-tuned) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense index | FAISS (`faiss-cpu`) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Answer LLM | `Qwen/Qwen2.5-1.5B-Instruct` (local Hugging Face / transformers) |
| Linguistic features | spaCy (`en_core_web_sm`) |
| Deep learning runtime | PyTorch (CUDA wheels on Linux via uv index) |

## Data

| Dataset | Role |
|---|---|
| UCI SMS Spam Collection | Binary ham/spam source → `data/raw/sms_spam.csv` |
| Super SMS Dataset | Binary labels → `data/raw/super_sms_dataset.csv` |
| Merged processed splits | `data/processed/messages_{train,val,test,all}.csv` |
| Knowledge corpus | ≥20 official FTC/CISA/SSA/… JSON docs in `data/knowledge/` |

## Tooling

| Tool | Use |
|---|---|
| **uv** | `uv sync` / `uv.lock` / `pyproject.toml` project env |
| Node / npm | Angular UI |
| matplotlib / wordcloud / plotly | Dashboard viz artifacts |

## Key Python packages (from `pyproject.toml`)

fastapi, uvicorn, pydantic, pandas, numpy, scikit-learn, joblib, torch, transformers, accelerate, sentence-transformers, faiss-cpu, rank-bm25, spacy, boto3, beautifulsoup4, httpx, matplotlib, seaborn, plotly, wordcloud
