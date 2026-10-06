"""NLP backends for sentiment scoring and event classification.

Hybrid design: tries pretrained transformer models first (FinBERT for
sentiment, a small zero-shot classifier for event category); if the
`transformers`/`torch` stack is unavailable or fails to load, permanently
falls back to a rule-based path (VADER + keyword dictionary) for the rest
of the run. Every signal is tagged with which method actually produced it
(see schemas.RiskSignal.method) -- no silent partial failures.
"""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SENTIMENT_MODEL_ID = "ProsusAI/finbert"
EVENT_MODEL_ID = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"

EVENT_LABELS = [
    "Geopolitical", "Macroeconomic", "Credit Event",
    "Merger/Acquisition", "Product Launch", "Other",
]

KEYWORD_RULES = {
    "Geopolitical": ["sanction", "war", "conflict", "geopolitic", "tension", "tariff", "ceasefire", "regional"],
    "Credit Event": ["default", "bankrupt", "downgrade", "credit", "covenant", "debt", "rating agency"],
    "Merger/Acquisition": ["acquire", "acquisition", "merger", "takeover", "buyout", "deal"],
    "Product Launch": ["launch", "unveil", "product", "showcase", "release"],
    "Macroeconomic": ["inflation", "interest rate", "gdp", "recession", "jobs data", "fed", "central bank"],
}

_vader = SentimentIntensityAnalyzer()

_sentiment_pipeline = None
_sentiment_backend_failed = False
_event_pipeline = None
_event_backend_failed = False


def _try_load_sentiment_pipeline():
    global _sentiment_pipeline, _sentiment_backend_failed
    if _sentiment_pipeline is not None or _sentiment_backend_failed:
        return _sentiment_pipeline
    try:
        from transformers import pipeline
        _sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
    except Exception:
        _sentiment_backend_failed = True
        _sentiment_pipeline = None
    return _sentiment_pipeline


def _try_load_event_pipeline():
    global _event_pipeline, _event_backend_failed
    if _event_pipeline is not None or _event_backend_failed:
        return _event_pipeline
    try:
        from transformers import pipeline
        _event_pipeline = pipeline("zero-shot-classification", model=EVENT_MODEL_ID)
    except Exception:
        _event_backend_failed = True
        _event_pipeline = None
    return _event_pipeline


def score_sentiment(text: str) -> tuple[float, float, str]:
    """Returns (sentiment_score in [-1,1], confidence in [0,1], method)."""
    pipe = _try_load_sentiment_pipeline()
    if pipe is not None:
        try:
            result = pipe(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            signed = score if label == "positive" else (-score if label == "negative" else 0.0)
            return round(signed, 4), round(score, 4), "transformer"
        except Exception:
            pass  # fall through to rule-based for this item only
    vs = _vader.polarity_scores(text)
    compound = vs["compound"]
    confidence = min(1.0, abs(compound) + 0.3)
    return round(compound, 4), round(confidence, 4), "rule_based"


def classify_event(text: str) -> tuple[str, float, str]:
    """Returns (event_classification, confidence in [0,1], method)."""
    pipe = _try_load_event_pipeline()
    if pipe is not None:
        try:
            result = pipe(text, candidate_labels=EVENT_LABELS)
            return result["labels"][0], round(result["scores"][0], 4), "transformer"
        except Exception:
            pass
    lowered = text.lower()
    best_category, best_hits = "Other", 0
    for category, keywords in KEYWORD_RULES.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_category, best_hits = category, hits
    confidence = min(1.0, 0.4 + 0.15 * best_hits) if best_hits else 0.3
    return best_category, round(confidence, 4), "rule_based"
