"""Heuristic 1-10 impact score.

Combines event-category severity, sentiment magnitude, classifier
confidence, and keyword-severity hits in the text. Deliberately a simple
weighted formula rather than a trained model -- the hackathon problem
statement explicitly permits a simplified severity model.
"""

SEVERITY_WEIGHTS = {
    "Credit Event": 0.90,
    "Geopolitical": 0.85,
    "Macroeconomic": 0.60,
    "Merger/Acquisition": 0.55,
    "Product Launch": 0.35,
    "Other": 0.30,
}

KEYWORD_BOOSTS = {
    "default": 1.0, "bankrupt": 1.0, "war": 1.0, "sanction": 0.9,
    "downgrade": 0.7, "crisis": 0.8, "shutdown": 0.7, "collapse": 0.7,
}


def compute_impact_score(event_classification: str, sentiment_score: float,
                          confidence: float, text: str) -> int:
    base = SEVERITY_WEIGHTS.get(event_classification, 0.3)
    magnitude = abs(sentiment_score)
    lowered = text.lower()
    keyword_boost = max((v for k, v in KEYWORD_BOOSTS.items() if k in lowered), default=0.0)

    raw = (0.45 * base) + (0.25 * magnitude) + (0.15 * confidence) + (0.15 * keyword_boost)
    score = 1 + round(raw * 9)
    return max(1, min(10, score))
