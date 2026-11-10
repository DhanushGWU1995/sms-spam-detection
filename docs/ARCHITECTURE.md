# ScamShield US layout

```
NLP_Project_Final/
  frontend/                 # Angular 19 + TypeScript UI (separate app)
  backend/app/              # FastAPI API + NLP + RAG + agents
  scripts/                  # data prep, train, index, S3 sync, pipeline runner
  notebooks/                # EDA / evaluation notebooks
  data/
    raw/ processed/ knowledge/ provenance/
  artifacts/
    models/ indexes/ metrics/
  configs/settings.yaml
  archive/sms_spam_v1/      # previous course prototype (source for UCI baseline)
```

S3 mirror:

`s3://dats-nlp/dhanush.mathivanan@gwu.edu/nlp-data/{raw,processed,knowledge,artifacts,provenance,backups}/`
