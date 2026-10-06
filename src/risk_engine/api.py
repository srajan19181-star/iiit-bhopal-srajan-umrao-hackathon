"""Optional FastAPI interface for the risk engine.

The submission spec requires signals be available "via a simple API or by
writing to a file" -- data/output/risk_signals.csv/.json (written by
run_batch.py) already satisfies this. This API is a bonus live-demo surface:
POST /analyze lets you score an arbitrary headline/tweet in real time.

Run from the repo root:
    uvicorn src.risk_engine.api:app --reload
"""
import json
from datetime import date as date_type
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .engine import analyze_text
from .schemas import RiskSignal

DATA_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "output"

app = FastAPI(title="AI/NLP Financial Risk Engine", version="1.0.0")


class AnalyzeRequest(BaseModel):
    text: str
    company: str = "Unknown"
    ticker: str = "N/A"
    source: str = "news"


@app.post("/analyze", response_model=RiskSignal)
def analyze(request: AnalyzeRequest) -> RiskSignal:
    return analyze_text(
        signal_id=f"api_{abs(hash(request.text)) % 100000}",
        text=request.text,
        company=request.company,
        ticker=request.ticker,
        source=request.source,
        date_=date_type.today(),
    )


@app.get("/signals")
def get_signals() -> list[dict]:
    path = DATA_OUTPUT / "risk_signals.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
