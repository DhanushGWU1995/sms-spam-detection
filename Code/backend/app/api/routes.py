from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.agents.workflow import investigate
from backend.app.analytics.metrics import (
    load_approach_reports,
    load_classification_metrics,
    load_dashboard_viz,
    load_knowledge_sources,
    load_llm_summary,
)
from backend.app.analytics.stats import load_summary, query_category
from backend.app.core.config import get_settings
from backend.app.nlp.classifier import ScamClassifier
from backend.app.nlp.features import extract_signals
from backend.app.rag.retriever import HybridRetriever

router = APIRouter()


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Suspicious SMS/email/message text")
    model: str = "auto"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    message: str = ""


@router.get("/health")
def health():
    settings = get_settings()
    retriever = HybridRetriever()
    classifier = ScamClassifier()
    return {
        "status": "ok",
        "project": settings.project_name,
        "llm": settings.llm_model,
        "embedding": settings.embedding_model,
        "classifier_models": classifier.available_models(),
        "rag_ready": retriever.ready(),
        "open_source_only": True,
    }


@router.post("/analyze")
def analyze(payload: AnalyzeRequest):
    classifier = ScamClassifier()
    result = classifier.predict(payload.message, model_name=payload.model)
    result["signals"] = extract_signals(payload.message).to_dict()
    return result


@router.post("/ask")
def ask(payload: AskRequest):
    return investigate(message=payload.message, question=payload.question)


@router.get("/stats")
def stats(category: str = ""):
    if category:
        return query_category(category)
    return load_summary()


@router.get("/models")
def models():
    settings = get_settings()
    return {
        "embedding": settings.embedding_model,
        "classifier_base": settings.classifier_model,
        "llm": settings.llm_model,
        "reranker": settings.reranker_model,
        "license_note": "All listed models are free open-source Hugging Face models. No paid API is used.",
        "available_classifiers": ScamClassifier().available_models(),
    }


@router.get("/metrics/classifiers")
def metrics_classifiers():
    return load_classification_metrics()


@router.get("/metrics/approaches")
def metrics_approaches():
    return load_approach_reports()


@router.get("/metrics/llm")
def metrics_llm():
    return load_llm_summary()


@router.get("/viz/overview")
def viz_overview():
    return load_dashboard_viz()


@router.get("/knowledge/sources")
def knowledge_sources():
    return load_knowledge_sources()
