"""Structured output schema for the AI/NLP Risk Engine."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

EventCategory = Literal[
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product Launch",
    "Other",
]


class RiskSignal(BaseModel):
    signal_id: str
    date: date
    source: Literal["news", "twitter"]
    company: str
    ticker: str
    text: str

    sentiment_score: float = Field(ge=-1.0, le=1.0)
    event_classification: EventCategory
    impact_score: int = Field(ge=1, le=10)

    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["transformer", "rule_based"]
    latency_ms: float
