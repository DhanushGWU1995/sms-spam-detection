# ScamShield US

Open-source agentic RAG assistant for consumer fraud / identity-theft education.

**Stack constraint:** AWS EC2 (`g5.xlarge`) + Amazon S3 only. No Bedrock, Lambda, API Gateway, RDS, OpenSearch, or SageMaker.

| Layer | Choice |
|---|---|
| Traditional classifiers | TF-IDF + Logistic Regression / Linear SVM |
| Advanced classifier | DistilBERT |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | Hybrid BM25 + FAISS (`top_k=5`) |
| Answer LLM | Qwen2.5-1.5B-Instruct (local on EC2) |
| Agents | Deterministic supervisor pipeline in FastAPI |
| API / UI | FastAPI + Angular TypeScript |
| Storage | S3 for all data/models/indexes via **boto3 only** (EC2 caches ephemerally; no AWS CLI) |
| Python env | **uv sync** from `pyproject.toml` + `uv.lock` → `.venv` (`scripts/setup_venv.sh`) |

---

## Quick start (new EC2)

```bash
# 1) Clone
cd ~
git clone https://github.com/DhanushGWU1995/sms-spam-detection.git nlp_project_final
cd nlp_project_final

# 2) AWS credentials (course SSO / STS)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-east-1"

# 3) uv sync from pyproject.toml / uv.lock → .venv (all Python deps)
bash scripts/setup_venv.sh
source .venv/bin/activate
export PYTHONPATH="$PWD"

# 4a) If S3 already has models/index/metrics (this course prefix does): skip rebuild
# 4b) Otherwise rebuild everything into S3:
# bash scripts/run_pipeline.sh

# 5) API
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# 6) UI (other terminal)
cd frontend && npm install && npm start
# http://127.0.0.1:4200  (or SSH tunnel from your laptop)
```

`setup_venv.sh` installs `uv` if needed, then runs **`uv sync`** against `pyproject.toml` + `uv.lock` (and refreshes `requirements.txt` via `uv export`). Linux x86_64 resolves CUDA PyTorch (`cu124`) from the index declared in `pyproject.toml`.

---

## Layout

```
backend/app/      FastAPI, NLP, RAG, investigation workflow
frontend/         Angular UI (proxies /api → :8000)
scripts/          setup_venv (uv sync) / pipeline / S3 helpers
configs/          settings.yaml (includes s3_uri)
docs/             architecture PPT, tech stack, project report
pyproject.toml    project metadata + dependency list (source of truth for uv)
uv.lock           locked versions for reproducible uv sync
requirements.txt  exported from uv.lock (convenience / readers)
data/             .gitkeep placeholders only (real data lives in S3)
artifacts/        .gitkeep placeholders only (real models/indexes live in S3)
.venv/            created by uv sync (gitignored)
```

Architecture materials: see [`docs/`](docs/) — especially `ScamShield_Architecture.pptx`, `ARCHITECTURE.md`, `TECH_STACK.md`, and `PROJECT_REPORT.md`.


## S3 layout (durable store)

```
s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/
  data/{raw,processed,knowledge,provenance}/
  artifacts/{models,indexes,metrics,viz}/
```

- **EC2 = code only.** Durable outputs are written to / read from S3 through `backend.app.core.s3_store.S3DataStore` (boto3).
- Ephemeral cache: `/tmp/scamshield-s3-cache` (override with `SCAMSHIELD_CACHE_DIR`).
- Do **not** use the AWS CLI for project data.

---

## Correct end-to-end flow

Run steps **in this order**. Steps 3–8 are also wrapped by `scripts/run_pipeline.sh`.

```
1. Clone / sync code on EC2
2. AWS credentials (STS / env vars)
3. Python env via uv sync      →  scripts/setup_venv.sh  (`pyproject.toml` / `uv.lock` → `.venv`)
4. Seed knowledge base         →  S3 data/knowledge + provenance
5. Prepare baseline splits     →  S3 data/processed
6. Train classifiers           →  S3 artifacts/models
7. Build RAG index             →  S3 artifacts/indexes
8. Evaluate (+ optional viz)   →  S3 artifacts/metrics (+ viz)
9. Start FastAPI backend       →  :8000
10. Start Angular UI           →  :4200 (local browser / SSH tunnel)
```

---

## 0) Prerequisites

- EC2 with GPU recommended (course `g5.xlarge`); CPU works for API-only if models already exist in S3
- SSH key (example): `AWS_NLP_Config_keys.pem`
- Temporary AWS credentials with S3 access to your course prefix
- Node.js 18+ where you run the Angular UI (EC2 or laptop)
- Network access so `uv` can install packages / PyTorch

Repo: [https://github.com/DhanushGWU1995/sms-spam-detection](https://github.com/DhanushGWU1995/sms-spam-detection)

---

## 1) Get the code on EC2

```bash
# from your laptop
ssh -i /path/to/AWS_NLP_Config_keys.pem ubuntu@<EC2_PUBLIC_IP>

# on EC2
cd ~
git clone https://github.com/DhanushGWU1995/sms-spam-detection.git nlp_project_final
# or, if already cloned:
cd ~/nlp_project_final && git pull origin main
```

Working directory for all later commands:

```bash
cd ~/nlp_project_final
```

---

## 2) Export AWS credentials (boto3)

Session tokens expire — refresh them from the course SSO portal when needed.

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-east-1"
```

Optional overrides (defaults come from `configs/settings.yaml`):

```bash
export SCAMSHIELD_S3_URI="s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/"
export SCAMSHIELD_CACHE_DIR="/tmp/scamshield-s3-cache"
```

---

## 3) Install Python deps with **`uv sync`** (fresh EC2)

Dependencies live in **`pyproject.toml`**. Locked versions are in **`uv.lock`**.  
`requirements.txt` is an export of that lock (convenience only; do not edit it by hand).

**Recommended (one command):**

```bash
cd ~/nlp_project_final
bash scripts/setup_venv.sh
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

What that script does:

1. Installs [`uv`](https://github.com/astral-sh/uv) into `~/.local/bin` if needed
2. `uv python install 3.11`
3. **`uv sync --python 3.11`** — creates `.venv` and installs from `pyproject.toml` / `uv.lock`
4. `uv export … -o requirements.txt` — refreshes the exported requirements file
5. `python -m spacy download en_core_web_sm`
6. Smoke-imports torch / fastapi / faiss / boto3 / spacy

**Manual equivalent:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.11
uv sync --python 3.11          # pyproject.toml + uv.lock → .venv
source .venv/bin/activate
python -m spacy download en_core_web_sm
export PYTHONPATH="$PWD"
```

After changing dependencies in `pyproject.toml`:

```bash
uv lock
uv sync
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
```

Activate in **every new shell** before pipeline / API:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

`scripts/run_pipeline.sh` prefers `.venv/bin/python` when present.

Quick S3 check:

```bash
python scripts/sync_s3.py --list
```

Force CPU torch on Linux (reinstall PyPI torch after sync):

```bash
SKIP_CUDA_TORCH=1 bash scripts/setup_venv.sh
```

---

## 4–8) Run the full data/model pipeline (S3 in / S3 out)

### Option A — one command (recommended)

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
bash scripts/run_pipeline.sh
```

This runs, in order:

| Step | Script | Writes to S3 |
|---|---|---|
| Init prefixes | `scripts/sync_s3.py --init` | empty `data/*` + `artifacts/*` markers |
| Seed KB (≥20 URL-backed docs) | `scripts/seed_knowledge_base.py` | `data/knowledge/`, `data/provenance/` |
| Prepare train/val/test splits | `scripts/prepare_baseline_data.py` | `data/processed/` |
| Train Traditional + DistilBERT | `scripts/train_classifiers.py` | `artifacts/models/` |
| Build hybrid BM25+FAISS index | `scripts/build_rag_index.py` | `artifacts/indexes/` |
| Evaluate approaches | `scripts/evaluate_approaches.py` | `artifacts/metrics/` |
| Dashboard figures | `scripts/build_dashboard_viz.py` | `artifacts/viz/` |

If S3 already has models/indexes/metrics (this course prefix does), you can skip the pipeline and go straight to the API.

### Option B — run steps individually

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"

python scripts/sync_s3.py --init
python scripts/seed_knowledge_base.py          # knowledge + provenance → S3
python scripts/prepare_baseline_data.py        # processed CSVs → S3
python scripts/train_classifiers.py            # sklearn + DistilBERT → S3
python scripts/build_rag_index.py              # FAISS + chunks → S3
python scripts/evaluate_approaches.py          # metrics → S3
python scripts/build_dashboard_viz.py          # plots → S3
```

After a successful pipeline, confirm:

```bash
python scripts/sync_s3.py --list
# expect artifacts/{indexes,models,metrics,viz} and data/{raw,knowledge,processed,provenance}
```

---

## 9) Start the FastAPI backend

Keep AWS credentials exported in this shell. The API downloads models/indexes from S3 into the ephemeral cache on first use.

```bash
cd ~/nlp_project_final
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Health check (from EC2 or via tunnel):

```bash
curl -s http://127.0.0.1:8000/api/health
# or open http://<EC2_PUBLIC_IP>:8000/docs
```

---

## 10) Start the Angular UI

### On EC2 (UI served from the instance)

```bash
cd ~/nlp_project_final/frontend
npm install
npm start
# ng serve on :4200, proxy /api → http://127.0.0.1:8000
```

Open from your laptop with an SSH tunnel:

```bash
# laptop
ssh -i /path/to/AWS_NLP_Config_keys.pem -L 4200:127.0.0.1:4200 -L 8000:127.0.0.1:8000 ubuntu@<EC2_PUBLIC_IP>
# then browse http://127.0.0.1:4200
```

### Or run UI locally against the EC2 API

1. Tunnel only the API: `ssh -i ... -L 8000:127.0.0.1:8000 ubuntu@<EC2_PUBLIC_IP>`
2. On your laptop:

```bash
cd frontend
npm install
npm start
# http://127.0.0.1:4200  (proxy.conf.json → http://127.0.0.1:8000)
```

---

## Runtime behavior (what the UI does)

1. User submits a message / case.
2. **Traditional / Advanced / Hybrid** classifiers score the text (models from S3).
3. Hybrid **BM25 + FAISS** retrieves the top **5** knowledge chunks from the S3-backed index.
4. Local **Qwen2.5-1.5B** answers using only those retrieved chunks (answer-only RAG).

---

## Useful boto3 helpers

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"

python scripts/sync_s3.py --list
python scripts/sync_s3.py --list data/knowledge/
python scripts/sync_s3.py --download          # pull data+artifacts into cache
python scripts/sync_s3.py --rm path/in/prefix  # delete one key
python scripts/sync_s3.py --recursive --rm unwanted/prefix/
```

---

## Local laptop notes

- Full GPU training / Qwen inference is intended on EC2.
- You can develop the Angular UI locally (step 10) while the API runs on EC2.
- On a laptop: `bash scripts/setup_venv.sh` (runs `uv sync` from `pyproject.toml` / `uv.lock`), export AWS credentials, then `source .venv/bin/activate`. S3 remains the source of truth.

---

## Safety

Educational decision support only. Not a legal or law-enforcement determination. The system does not contact senders, submit reports, or access financial accounts.
