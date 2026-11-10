"""Indicator extraction for suspicious consumer messages."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?|\b\d+\s?(?:percent|%)\b", re.I)
SSN_RE = re.compile(r"\b(?:ssn|social security)\b", re.I)

URGENCY = (
    "urgent",
    "immediately",
    "suspended",
    "locked",
    "verify now",
    "act now",
    "final notice",
    "within 24 hours",
)
CREDENTIAL = (
    "password",
    "pin",
    "one-time code",
    "otp",
    "account number",
    "routing number",
    "social security",
    "ssn",
    "date of birth",
)
PAYMENT = (
    "gift card",
    "wire transfer",
    "bitcoin",
    "crypto",
    "zelle",
    "venmo",
    "western union",
    "cash app",
)
IMPERSONATION = (
    "irs",
    "social security",
    "ssa",
    "amazon",
    "paypal",
    "chase",
    "bank of america",
    "wells fargo",
    "microsoft",
    "apple support",
    "fedex",
    "ups",
)


@dataclass
class MessageSignals:
    urls: list[str]
    emails: list[str]
    phones: list[str]
    money_mentions: list[str]
    urgency_cues: list[str]
    credential_requests: list[str]
    payment_methods: list[str]
    impersonated_orgs: list[str]
    asks_for_sensitive_info: bool
    score_hints: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _hits(text: str, lexicon: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    return sorted({term for term in lexicon if term in lower})


def extract_signals(text: str) -> MessageSignals:
    text = text or ""
    urls = URL_RE.findall(text)
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    money = MONEY_RE.findall(text)
    urgency = _hits(text, URGENCY)
    credential = _hits(text, CREDENTIAL)
    payment = _hits(text, PAYMENT)
    orgs = _hits(text, IMPERSONATION)
    sensitive = bool(credential or SSN_RE.search(text))
    hints = {
        "urgency": min(1.0, 0.2 * len(urgency)),
        "sensitive_request": 0.35 if sensitive else 0.0,
        "payment_pressure": min(1.0, 0.25 * len(payment)),
        "link_present": 0.2 if urls else 0.0,
        "impersonation_lexicon": min(1.0, 0.2 * len(orgs)),
    }
    return MessageSignals(
        urls=urls,
        emails=emails,
        phones=phones,
        money_mentions=money,
        urgency_cues=urgency,
        credential_requests=credential,
        payment_methods=payment,
        impersonated_orgs=orgs,
        asks_for_sensitive_info=sensitive,
        score_hints=hints,
    )
