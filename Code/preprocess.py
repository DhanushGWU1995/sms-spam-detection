"""Cleaning for the linear model, the GRU, and the Streamlit app.

DistilBERT does not use this. It tokenizes the raw message itself.
"""

import html
import re

# Only drop function words. "free", "call", and "txt" are spam clues, so they stay.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "is", "it", "this", "that", "with", "as", "be", "by",
    "from", "are", "was", "were", "been", "am",
}


def preprocess_text(text: str) -> str:
    # The file stores things like &lt;#&gt; and &lt;URL&gt; instead of the real number or link.
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\S+", " url ", text)
    text = (
        text.replace("<#>", " num ")
        .replace("<url>", " url ")
        .replace("<time>", " time ")
        .replace("<decimal>", " decimal ")
        .replace("<email>", " email ")
    )
    # Keep digits and currency. Spam often has a phone number or a price.
    text = re.sub(r"[^a-z0-9£$\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in STOPWORDS]
    return " ".join(tokens) if tokens else "empty"
