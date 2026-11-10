"""Supervisor-style agent workflow for fraud investigation."""

from __future__ import annotations

from typing import Any, TypedDict

from backend.app.analytics.stats import query_category
from backend.app.nlp.classifier import ScamClassifier
from backend.app.rag.llm import generate_answer
from backend.app.rag.retriever import HybridRetriever


class InvestigationState(TypedDict, total=False):
    message: str
    question: str
    classification: dict[str, Any]
    retrieval: list[dict[str, Any]]
    stats: dict[str, Any]
    draft: dict[str, Any]
    validation: dict[str, Any]
    trace: list[str]


_classifier: ScamClassifier | None = None
_retriever: HybridRetriever | None = None


def _get_classifier() -> ScamClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ScamClassifier()
    return _classifier


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _validate(draft: dict[str, Any], retrieval: list[dict[str, Any]]) -> dict[str, Any]:
    answer = draft.get("answer", "")
    missing_citations = bool(retrieval) and "[" not in answer
    unsupported_certainty = any(
        phrase in answer.lower()
        for phrase in ("definitely a scam", "proven fraud", "guaranteed", "the irs contacted you")
    )
    warnings = []
    if missing_citations:
        warnings.append("Model response did not include bracket citations; UI will still show source cards.")
    if unsupported_certainty:
        warnings.append("Response contained overly certain language; treat as educational only.")
    if not retrieval:
        warnings.append("No official passages retrieved; answer may be under-grounded.")
    return {
        "ok": not unsupported_certainty,
        "warnings": warnings,
        "evidence_count": len(retrieval),
    }


def investigate(message: str = "", question: str = "") -> dict[str, Any]:
    """Run the specialized-agent pipeline end to end."""
    state: InvestigationState = {
        "message": message or "",
        "question": question or message or "What should I do about this message?",
        "trace": [],
    }

    # Fraud Detection Agent
    state["trace"].append("supervisor: route to fraud_detection")
    classification = _get_classifier().predict(message or question)
    state["classification"] = classification
    state["trace"].append(f"fraud_detection: label={classification['label']} conf={classification['confidence']:.3f}")

    # Data Intelligence Agent
    state["trace"].append("supervisor: route to data_intelligence")
    keyword = classification["label"].replace("_", " ")
    stats = query_category(keyword.split()[0] if keyword else "")
    state["stats"] = stats
    state["trace"].append(f"data_intelligence: matches={len(stats.get('matches', []))}")

    # Knowledge Retrieval Agent
    state["trace"].append("supervisor: route to knowledge_retrieval")
    query = question or message
    retrieval = _get_retriever().search(query, top_k=5, mode="hybrid")
    state["retrieval"] = retrieval
    state["trace"].append(f"knowledge_retrieval: hits={len(retrieval)}")

    # Consumer Assistance Agent (local OSS LLM)
    state["trace"].append("supervisor: route to consumer_assistance")
    consumer_question = (
        f"Submitted message:\n{message}\n\n"
        f"Classifier label: {classification['label']} "
        f"(confidence {classification['confidence']:.2f})\n"
        f"User question: {state['question']}"
    )
    try:
        draft = generate_answer(consumer_question, retrieval)
    except Exception as exc:  # pragma: no cover - depends on local GPU/weights
        from backend.app.rag.llm import build_llm_messages

        system, user, messages, context = build_llm_messages(consumer_question, retrieval)
        draft = {
            "model": "fallback-template",
            "answer": (
                "I could not load the local open-source LLM on this machine. "
                "Based on the classifier and retrieved guidance cards below, do not click links, "
                "do not share passwords or Social Security numbers, and use official websites you "
                f"type yourself. Predicted category: {classification['label']}. "
                f"Technical detail: {exc}"
            ),
            "llm_input": {
                "model": "fallback-template",
                "system": system,
                "user": user,
                "messages": messages,
                "chat_template_prompt": None,
                "retrieved_passage_count": len(retrieval),
                "context_preview": context[:4000],
                "note": f"Local Qwen load failed; showing the prompt that would have been sent. ({exc})",
                "generation": {"max_new_tokens": 320, "do_sample": False},
            },
            "citations": [
                {
                    "n": i + 1,
                    "title": block.get("title"),
                    "agency": block.get("agency"),
                    "source_url": block.get("source_url"),
                    "retrieved_date": block.get("retrieved_date"),
                }
                for i, block in enumerate(retrieval)
            ],
        }
    state["draft"] = draft
    state["trace"].append(f"consumer_assistance: model={draft.get('model')}")

    # Evidence Validator Agent
    state["trace"].append("supervisor: route to evidence_validator")
    validation = _validate(draft, retrieval)
    state["validation"] = validation
    state["trace"].append(f"evidence_validator: ok={validation['ok']}")

    llm_input = draft.get("llm_input") or {
        "model": draft.get("model"),
        "consumer_question_bundle": consumer_question,
        "retrieved_passage_count": len(retrieval),
    }
    llm_input["consumer_question_bundle"] = consumer_question
    llm_input["classification"] = {
        "label": classification.get("label"),
        "confidence": classification.get("confidence"),
        "model": classification.get("model"),
    }
    llm_input["user_question"] = state["question"]
    llm_input["submitted_message"] = message

    return {
        "classification": classification,
        "stats": stats,
        "retrieval": retrieval,
        "answer": draft.get("answer"),
        "model": draft.get("model"),
        "citations": draft.get("citations", []),
        "llm_input": llm_input,
        "validation": validation,
        "trace": state["trace"],
        "guardrails": {
            "no_account_actions": True,
            "educational_only": True,
            "read_only_sources": True,
        },
    }
