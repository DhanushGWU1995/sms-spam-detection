"""Traditional and DistilBERT classifiers — open-source only."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store
from backend.app.nlp.features import extract_signals

LABELS = [
    "legitimate",
    "phishing_spoofing",
    "government_impersonation",
    "banking_impersonation",
    "investment_scam",
    "identity_theft_attempt",
    "other_uncertain",
]


class ScamClassifier:
    def __init__(self) -> None:
        settings = get_settings()
        self.models_key = settings.models_key.rstrip("/")
        self.bundle: dict[str, Any] | None = None
        self.distilbert = None
        self._load()

    def _load(self) -> None:
        store = get_store()
        bundle_rel = f"{self.models_key}/sklearn_bundle.joblib"
        labels_rel = f"{self.models_key}/label_list.json"
        bert_rel = f"{self.models_key}/distilbert_scam"

        if store.exists(bundle_rel):
            path = store.ensure_local(bundle_rel)
            self.bundle = joblib.load(path)
            # Drop MultinomialNB from older bundles without retraining.
            models = dict(self.bundle.get("models") or {})
            models.pop("multinomial_nb", None)
            self.bundle["models"] = models
            tfidf_only = [
                name
                for name in (self.bundle.get("tfidf_only_models") or [])
                if name != "multinomial_nb"
            ]
            self.bundle["tfidf_only_models"] = tfidf_only
        if store.exists(labels_rel):
            self.labels = store.read_json(labels_rel)
        else:
            self.labels = LABELS
        if store.exists(f"{bert_rel}/config.json"):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            bert_dir = store.ensure_prefix_local(bert_rel)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            self.distilbert = AutoModelForSequenceClassification.from_pretrained(bert_dir)
            self.distilbert.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.distilbert.to(self.device)

    def available_models(self) -> list[str]:
        names = []
        if self.bundle:
            names.extend(sorted(self.bundle.get("models", {}).keys()))
        if self.distilbert is not None:
            names.append("distilbert")
        return names

    def _rule_prior(self, text: str) -> dict[str, float]:
        signals = extract_signals(text)
        hints = signals.score_hints
        risk = sum(hints.values())
        prior = {label: 0.05 for label in self.labels}
        if risk < 0.15 and not signals.urls:
            prior["legitimate"] = 0.7
        elif any(org in ("irs", "ssa", "social security") for org in signals.impersonated_orgs):
            prior["government_impersonation"] = 0.55
        elif any(org in ("chase", "bank of america", "wells fargo", "paypal") for org in signals.impersonated_orgs):
            prior["banking_impersonation"] = 0.5
        elif signals.money_mentions and ("percent" in " ".join(signals.money_mentions).lower() or "%" in " ".join(signals.money_mentions)):
            prior["investment_scam"] = 0.5
        elif signals.asks_for_sensitive_info:
            prior["identity_theft_attempt"] = 0.45
        elif signals.urls or signals.urgency_cues:
            prior["phishing_spoofing"] = 0.45
        else:
            prior["other_uncertain"] = 0.4
        total = sum(prior.values())
        return {k: v / total for k, v in prior.items()}

    def predict(self, text: str, model_name: str = "auto") -> dict[str, Any]:
        signals = extract_signals(text)
        chosen = model_name
        proba_map: dict[str, float]

        if model_name == "auto":
            if self.distilbert is not None:
                chosen = "distilbert"
            elif self.bundle and "logistic_regression" in self.bundle.get("models", {}):
                chosen = "logistic_regression"
            else:
                chosen = "rules"

        if chosen == "distilbert" and self.distilbert is not None:
            import torch

            encoded = self.tokenizer(
                text, truncation=True, padding="max_length", max_length=160, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.distilbert(**encoded).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            id2label = {int(k): v for k, v in self.distilbert.config.id2label.items()}
            proba_map = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        elif chosen in (self.bundle or {}).get("models", {}):
            from scipy.sparse import csr_matrix, hstack

            from backend.app.nlp.domain_lexicon import extract_linguistic_matrix

            vectorizer = self.bundle["vectorizer"]
            model = self.bundle["models"][chosen]
            tfidf = vectorizer.transform([text])
            tfidf_only = set(self.bundle.get("tfidf_only_models") or [])
            if chosen in tfidf_only:
                features = tfidf
            elif "scaler" in self.bundle or "linguistic_feature_names" in self.bundle:
                ling = csr_matrix(extract_linguistic_matrix([text]))
                features = hstack([tfidf, ling]).tocsr()
                scaler = self.bundle.get("scaler")
                if scaler is not None:
                    features = scaler.transform(features)
            else:
                features = tfidf
            probs = model.predict_proba(features)[0]
            classes = list(model.classes_)
            proba_map = {str(c): float(p) for c, p in zip(classes, probs)}
        else:
            chosen = "rules"
            proba_map = self._rule_prior(text)

        label = max(proba_map, key=proba_map.get)
        confidence = float(proba_map[label])
        ranked = sorted(proba_map.values(), reverse=True)
        top_margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else confidence
        from backend.app.analytics.metrics import weak_confidence_margin

        margin = weak_confidence_margin()
        escalate_to_advanced = bool(top_margin <= margin)
        return {
            "model": chosen,
            "label": label,
            "confidence": confidence,
            "top_margin": top_margin,
            "weak_confidence_margin": margin,
            "escalate_to_advanced": escalate_to_advanced,
            "llm_would_trigger": False,
            "probabilities": proba_map,
            "signals": signals.to_dict(),
            "disclaimer": (
                "This is an automated risk assessment for education only. "
                "It is not proof that the sender is or is not engaging in fraud."
            ),
        }
