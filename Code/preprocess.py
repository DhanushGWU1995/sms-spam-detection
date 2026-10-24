"""Shared SMS cleaning for the classical model, the GRU, and the Streamlit app.

DistilBERT does not use this function. It tokenizes the raw message.
"""

import html
import re

# Short function words only. SMS tokens such as "free", "call", "txt",
# and "claim" are spam evidence and stay in the text.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "is", "it", "this", "that", "with", "as", "be", "by",
    "from", "are", "was", "were", "been", "am",
}


def preprocess_text(text: str) -> str:
    # The corpus stores redacted spans as HTML entities, e.g. &lt;#&gt; and &lt;URL&gt;.
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\S+", " url ", text)
    text = (
        text.replace("<#>", " num ")
        .replace("<url>", " url ")
        .replace("<time>", " time ")
        .replace("<decimal>", " decimal ")
        .replace("<email>", " email ")
    )
    # Keep digits and currency. Short codes, prices, and amounts are spam signal.
    text = re.sub(r"[^a-z0-9£$\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in STOPWORDS]
    return " ".join(tokens) if tokens else "empty"
