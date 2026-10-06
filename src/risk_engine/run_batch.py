"""CLI entrypoint: processes all bundled data and writes data/output/*.

Run from the repo root:
    python -m src.risk_engine.run_batch
"""
import json
import time
from pathlib import Path

import pandas as pd

from .engine import run_engine_on_dataframe
from .ingestion import unify_sources

DATA_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "output"


def _sentiment_bucket(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def build_eval_summary(df: pd.DataFrame, signals_df: pd.DataFrame, elapsed_s: float) -> dict:
    merged = signals_df.merge(df[["id", "intended_sentiment", "intended_event"]],
                               left_on="signal_id", right_on="id", how="left")

    merged["sentiment_bucket"] = merged["sentiment_score"].apply(_sentiment_bucket)
    sentiment_matches = (merged["sentiment_bucket"] == merged["intended_sentiment"]).mean()
    event_matches = (merged["event_classification"] == merged["intended_event"]).mean()

    by_method = {}
    for method, group in merged.groupby("method"):
        by_method[method] = {
            "n_signals": int(len(group)),
            "mean_latency_ms": round(float(group["latency_ms"].mean()), 3),
            "sentiment_self_consistency": round(float((group["sentiment_bucket"] == group["intended_sentiment"]).mean()), 4),
            "event_self_consistency": round(float((group["event_classification"] == group["intended_event"]).mean()), 4),
        }

    return {
        "note": ("Self-consistency compares the engine's output against the intended "
                 "sentiment/event label baked into our own synthetic data generator "
                 "(scripts/generate_synthetic_data.py). This is a reproducibility/sanity "
                 "check, NOT a real-world benchmark against human-labeled ground truth."),
        "n_total_signals": int(len(merged)),
        "overall_sentiment_self_consistency": round(float(sentiment_matches), 4),
        "overall_event_self_consistency": round(float(event_matches), 4),
        "n_high_impact_signals_gt7": int((merged["impact_score"] > 7).sum()),
        "method_distribution": merged["method"].value_counts().to_dict(),
        "by_method": by_method,
        "total_batch_runtime_seconds": round(elapsed_s, 3),
    }


def main():
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    df = unify_sources()

    start = time.perf_counter()
    signals = run_engine_on_dataframe(df)
    elapsed_s = time.perf_counter() - start

    signals_df = pd.DataFrame([s.model_dump() for s in signals])
    signals_df.to_csv(DATA_OUTPUT / "risk_signals.csv", index=False)
    (DATA_OUTPUT / "risk_signals.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in signals], indent=2, default=str), encoding="utf-8"
    )

    summary = build_eval_summary(df, signals_df, elapsed_s)
    (DATA_OUTPUT / "engine_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Processed {len(signals)} signals in {elapsed_s:.2f}s.")
    print(f"Method distribution: {summary['method_distribution']}")
    print(f"Overall sentiment self-consistency: {summary['overall_sentiment_self_consistency']:.1%}")
    print(f"Overall event self-consistency: {summary['overall_event_self_consistency']:.1%}")
    print(f"High-impact (>7) signals: {summary['n_high_impact_signals_gt7']}")
    print(f"Wrote outputs to {DATA_OUTPUT}")


if __name__ == "__main__":
    main()
