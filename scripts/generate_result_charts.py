"""Generates docs/assets/*.png result charts from the actual pipeline output.

Run AFTER `python -m src.risk_engine.run_batch` from the repo root:
    python scripts/generate_result_charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.module_a_rebalancer.rebalance import backtest  # noqa: E402
from src.module_b_stress_test.stress_test import find_trigger_events, run_stress_test  # noqa: E402

DATA_RAW = ROOT / "data" / "raw"
DATA_OUTPUT = ROOT / "data" / "output"
ASSETS = ROOT / "docs" / "assets"


def chart_weight_drift(signals: pd.DataFrame):
    index_config = pd.read_csv(DATA_RAW / "mock_index.csv")
    prices = pd.read_csv(DATA_RAW / "mock_index_prices.csv", parse_dates=["date"])
    bt = backtest(prices, signals, index_config, k=0.1)

    weight_cols = [c for c in bt.columns if c.startswith("weight_")]
    tickers = [c.replace("weight_", "") for c in weight_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.stackplot(bt["date"], *[bt[c] for c in weight_cols], labels=tickers)
    ax1.set_title("Module A: Index Weights Over Time")
    ax1.set_ylabel("Weight")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    ax1.tick_params(axis="x", rotation=30)

    ax2.plot(bt["date"], bt["sentiment_tilted_value"], label="Sentiment-tilted index", linewidth=2)
    ax2.plot(bt["date"], bt["equal_weight_value"], label="Equal-weight baseline", linewidth=2, linestyle="--")
    ax2.set_title("Sentiment-Tilted vs. Equal-Weight Baseline")
    ax2.set_ylabel("Normalized value (start = 1.0)")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(ASSETS / "weight_drift.png", dpi=150, facecolor="white")
    plt.close(fig)

    final = bt.iloc[-1]
    delta_pct = (final["sentiment_tilted_value"] / final["equal_weight_value"] - 1) * 100
    print(f"Module A: sentiment-tilted index ended {delta_pct:+.2f}% vs. equal-weight baseline.")
    return delta_pct


def chart_stress_before_after(signals: pd.DataFrame):
    portfolio = pd.read_csv(DATA_RAW / "synthetic_portfolio.csv")
    shock_config = json.loads((DATA_RAW / "shock_config.json").read_text(encoding="utf-8"))

    triggers = find_trigger_events(signals, impact_threshold=7)
    if len(triggers) == 0:
        print("No trigger events found at threshold 7 -- skipping stress chart.")
        return None

    event = triggers.loc[triggers["impact_score"].idxmax()]
    result = run_stress_test(event, portfolio, shock_config)

    breakdown = pd.DataFrame(result["breakdown_by_asset_type"]).set_index("asset_type")

    fig, ax = plt.subplots(figsize=(8, 5))
    breakdown[["pre_value", "post_value"]].plot(kind="bar", ax=ax, color=["#8CBEDB", "#E8A2A2"])
    ax.set_title(f"Module B: Portfolio Stress Test\n{result['event_classification']} event, "
                 f"impact {result['impact_score']}/10 ({result['pct_change']:+.2f}% total)")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(["Before", "After"])
    fig.tight_layout()
    fig.savefig(ASSETS / "stress_before_after.png", dpi=150, facecolor="white")
    plt.close(fig)

    print(f"Module B: representative trigger '{result['event_classification']}' "
          f"-> {result['pct_change']:+.2f}% portfolio impact.")
    return result


def chart_self_consistency(summary: dict):
    by_method = summary.get("by_method", {})
    if not by_method:
        print("No by_method breakdown in engine_eval_summary.json -- skipping self-consistency chart.")
        return

    methods = list(by_method.keys())
    sentiment_acc = [by_method[m]["sentiment_self_consistency"] * 100 for m in methods]
    event_acc = [by_method[m]["event_self_consistency"] * 100 for m in methods]
    latency = [by_method[m]["mean_latency_ms"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(methods))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], sentiment_acc, width, label="Sentiment")
    ax1.bar([i + width / 2 for i in x], event_acc, width, label="Event category")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Self-consistency vs. synthetic intended label (%)")
    ax1.set_title("Accuracy by NLP method")
    ax1.legend()

    ax2.bar(methods, latency, color="#C9A6E0")
    ax2.set_ylabel("Mean latency (ms/item)")
    ax2.set_title("Speed by NLP method")

    fig.tight_layout()
    fig.savefig(ASSETS / "self_consistency.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Self-consistency chart written for methods: {methods}")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(DATA_OUTPUT / "risk_signals.csv", parse_dates=["date"])
    summary = json.loads((DATA_OUTPUT / "engine_eval_summary.json").read_text(encoding="utf-8"))

    chart_weight_drift(signals)
    chart_stress_before_after(signals)
    chart_self_consistency(summary)
    print(f"Wrote charts to {ASSETS}")


if __name__ == "__main__":
    main()
