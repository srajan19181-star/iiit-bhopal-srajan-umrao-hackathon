"""Orchestrates ingestion -> NLP -> impact scoring -> RiskSignal records."""
import time
from datetime import date as date_type

import pandas as pd

from .impact_scoring import compute_impact_score
from .nlp_models import classify_event, score_sentiment
from .schemas import RiskSignal


def analyze_text(signal_id: str, text: str, company: str, ticker: str,
                  source: str, date_: date_type) -> RiskSignal:
    start = time.perf_counter()
    sentiment_score, sent_confidence, sent_method = score_sentiment(text)
    event_classification, event_confidence, event_method = classify_event(text)
    impact_score = compute_impact_score(event_classification, sentiment_score, event_confidence, text)
    latency_ms = (time.perf_counter() - start) * 1000

    method = "transformer" if (sent_method == "transformer" or event_method == "transformer") else "rule_based"
    confidence = round((sent_confidence + event_confidence) / 2, 4)

    return RiskSignal(
        signal_id=signal_id,
        date=date_,
        source=source,
        company=company,
        ticker=ticker,
        text=text,
        sentiment_score=sentiment_score,
        event_classification=event_classification,
        impact_score=impact_score,
        confidence=confidence,
        method=method,
        latency_ms=round(latency_ms, 3),
    )


def run_engine_on_dataframe(df: pd.DataFrame) -> list[RiskSignal]:
    signals = []
    for row in df.itertuples(index=False):
        signal = analyze_text(
            signal_id=str(row.id),
            text=str(row.text),
            company=str(row.company),
            ticker=str(row.ticker),
            source=str(row.source),
            date_=row.date.date() if hasattr(row.date, "date") else row.date,
        )
        signals.append(signal)
    return signals
