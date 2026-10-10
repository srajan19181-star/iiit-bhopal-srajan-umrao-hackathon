"""Lightweight end-to-end smoke test.

Verifies that the Risk Engine, Module A, and Module B all produce sensible
output using the committed data files. No external network access required.

Run from the repo root:
    py scripts/smoke_test.py
"""
import io
import json
import sys
from datetime import date
from pathlib import Path

# Force UTF-8 output so box-drawing chars work on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
errors: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"{PASS}  {label}")
    else:
        msg = f"{FAIL}  {label}" + (f"  — {detail}" if detail else "")
        print(msg)
        errors.append(label)


# ── Risk Engine ──────────────────────────────────────────────────────────────
print("\n── Risk Engine ──")
try:
    from src.risk_engine.engine import analyze_text

    sig = analyze_text(
        "smoke_1",
        "Apple shares plunge as new sanctions target overseas operations amid escalating conflict.",
        "Apple Inc.", "AAPL", "news", date.today(),
    )
    check("Sentiment score in [-1, 1]", -1.0 <= sig.sentiment_score <= 1.0)
    check("Sentiment is negative", sig.sentiment_score < 0,
          f"got {sig.sentiment_score}")
    check("Impact score in [1, 10]", 1 <= sig.impact_score <= 10)
    check("Event classification non-empty", bool(sig.event_classification))
    check("Method tag present", sig.method in ("transformer", "rule_based"))
    check("Latency recorded", sig.latency_ms >= 0)
except Exception as exc:
    errors.append(f"Risk Engine import/call failed: {exc}")
    print(f"{FAIL}  Risk Engine raised: {exc}")

# ── Output files ─────────────────────────────────────────────────────────────
print("\n── Output files ──")
DATA_OUTPUT = ROOT / "data" / "output"
for fname in ("risk_signals.csv", "risk_signals.json", "engine_eval_summary.json"):
    path = DATA_OUTPUT / fname
    check(f"{fname} exists", path.exists())

signals = pd.read_csv(DATA_OUTPUT / "risk_signals.csv", parse_dates=["date"])
check("risk_signals.csv has rows", len(signals) > 0, f"got {len(signals)}")
check("All required columns present",
      {"sentiment_score", "event_classification", "impact_score"}.issubset(signals.columns))
check("Sentiment scores in range",
      signals["sentiment_score"].between(-1, 1).all())
check("Impact scores in range",
      signals["impact_score"].between(1, 10).all())

summary = json.loads((DATA_OUTPUT / "engine_eval_summary.json").read_text())
check("High-impact events exist", summary.get("n_high_impact_signals_gt7", 0) > 0,
      f"got {summary.get('n_high_impact_signals_gt7')}")

# ── Module A ─────────────────────────────────────────────────────────────────
print("\n── Module A: Index Rebalancer ──")
try:
    from src.module_a_rebalancer.rebalance import backtest

    DATA_RAW = ROOT / "data" / "raw"
    prices = pd.read_csv(DATA_RAW / "mock_index_prices.csv", parse_dates=["date"])
    idx_cfg = pd.read_csv(DATA_RAW / "mock_index.csv")
    bt = backtest(prices, signals, idx_cfg, k=0.1)
    check("Backtest returns rows", len(bt) > 0)
    check("Columns include portfolio values",
          {"sentiment_tilted_value", "equal_weight_value"}.issubset(bt.columns))
    weight_cols = [c for c in bt.columns if c.startswith("weight_")]
    last_weights = bt.iloc[-1][weight_cols]
    check("Weights sum to ~1", abs(last_weights.sum() - 1.0) < 1e-6,
          f"sum={last_weights.sum():.6f}")
    final_tilted = bt.iloc[-1]["sentiment_tilted_value"]
    final_eq = bt.iloc[-1]["equal_weight_value"]
    delta_pct = (final_tilted / final_eq - 1) * 100
    print(f"       sentiment-tilted vs equal-weight: {delta_pct:+.2f}%")
except Exception as exc:
    errors.append(f"Module A raised: {exc}")
    print(f"{FAIL}  Module A raised: {exc}")

# ── Module B ─────────────────────────────────────────────────────────────────
print("\n── Module B: Stress Tester ──")
try:
    from src.module_b_stress_test.stress_test import find_trigger_events, run_stress_test

    DATA_RAW = ROOT / "data" / "raw"
    portfolio = pd.read_csv(DATA_RAW / "synthetic_portfolio.csv")
    shock_cfg = json.loads((DATA_RAW / "shock_config.json").read_text())
    triggers = find_trigger_events(signals, impact_threshold=7)
    check("Trigger events found", len(triggers) > 0, f"got {len(triggers)}")
    event = triggers.iloc[0]
    result = run_stress_test(event, portfolio, shock_cfg)
    check("Stress test has pre/post values",
          result["pre_value"] > 0 and result["post_value"] > 0)
    check("Breakdown by asset type present",
          len(result["breakdown_by_asset_type"]) > 0)
    check("High-impact event causes loss", result["pct_change"] < 0,
          f"pct_change={result['pct_change']:+.2f}%")
    print(f"       worst trigger pct_change: {result['pct_change']:+.2f}%")
except Exception as exc:
    errors.append(f"Module B raised: {exc}")
    print(f"{FAIL}  Module B raised: {exc}")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91m{len(errors)} check(s) FAILED:\033[0m")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("\033[92mAll checks passed.\033[0m")
