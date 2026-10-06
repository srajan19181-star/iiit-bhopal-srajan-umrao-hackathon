"""Unified Streamlit dashboard: Risk Engine overview + Module A + Module B.

Run from the repo root:
    streamlit run src/app.py

Reads only the committed data/output/* and data/raw/* files -- no heavy
NLP dependencies are needed just to view this dashboard.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.module_a_rebalancer.rebalance import backtest  # noqa: E402
from src.module_b_stress_test.stress_test import find_trigger_events, run_stress_test  # noqa: E402

DATA_RAW = ROOT / "data" / "raw"
DATA_OUTPUT = ROOT / "data" / "output"

st.set_page_config(page_title="AI/NLP Financial Risk Engine", layout="wide")


@st.cache_data
def load_signals() -> pd.DataFrame:
    df = pd.read_csv(DATA_OUTPUT / "risk_signals.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_eval_summary() -> dict:
    return json.loads((DATA_OUTPUT / "engine_eval_summary.json").read_text(encoding="utf-8"))


@st.cache_data
def load_index_config() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "mock_index.csv")


@st.cache_data
def load_prices() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "mock_index_prices.csv", parse_dates=["date"])


@st.cache_data
def load_portfolio() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "synthetic_portfolio.csv")


@st.cache_data
def load_shock_config() -> dict:
    return json.loads((DATA_RAW / "shock_config.json").read_text(encoding="utf-8"))


st.title("AI/NLP Financial Risk Engine")
st.caption("S&P Global & Crisil Campus Hackathon 2026 — Srajan Umrao, IIIT Bhopal")

if not (DATA_OUTPUT / "risk_signals.csv").exists():
    st.error(
        "No risk signals found. Run the pipeline first:\n\n"
        "```\npython scripts/generate_synthetic_data.py\npython -m src.risk_engine.run_batch\n```"
    )
    st.stop()

signals = load_signals()
summary = load_eval_summary()

tab_overview, tab_a, tab_b = st.tabs([
    "Risk Engine Overview", "Module A: Index Rebalancer", "Module B: Stress Testing",
])

with tab_overview:
    st.subheader("Engine output at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total signals", summary["n_total_signals"])
    c2.metric("High-impact signals (>7)", summary["n_high_impact_signals_gt7"])
    c3.metric("Sentiment self-consistency", f"{summary['overall_sentiment_self_consistency']:.0%}")
    c4.metric("Event self-consistency", f"{summary['overall_event_self_consistency']:.0%}")

    st.caption(summary["note"])

    method_counts = pd.Series(summary["method_distribution"], name="count")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.write("**Signals by NLP method**")
        st.bar_chart(method_counts)
    with mc2:
        st.write("**Event classification breakdown**")
        st.bar_chart(signals["event_classification"].value_counts())

    if summary.get("by_method"):
        st.write("**Latency by method (ms/item)**")
        latency_df = pd.DataFrame({
            m: {"mean_latency_ms": v["mean_latency_ms"], "n_signals": v["n_signals"]}
            for m, v in summary["by_method"].items()
        }).T
        st.dataframe(latency_df, width="stretch")

    st.write("**Sentiment score distribution**")
    st.bar_chart(signals["sentiment_score"].round(1).value_counts().sort_index())

    st.write("**Risk signals**")
    source_filter = st.multiselect("Source", options=sorted(signals["source"].unique()),
                                    default=sorted(signals["source"].unique()))
    filtered = signals[signals["source"].isin(source_filter)]
    st.dataframe(
        filtered[["date", "source", "company", "ticker", "sentiment_score",
                  "event_classification", "impact_score", "method", "text"]]
        .sort_values("date", ascending=False),
        width="stretch", height=400,
    )

with tab_a:
    st.subheader("Tactical sentiment-driven index rebalancing")
    st.write(
        "Each day, every stock's weight is nudged up on positive sentiment and down on "
        "negative sentiment, then renormalized. Compared against a static equal-weight baseline."
    )
    k = st.slider("Sentiment sensitivity (k)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

    index_config = load_index_config()
    prices = load_prices()
    bt = backtest(prices, signals, index_config, k=k)

    weight_cols = [c for c in bt.columns if c.startswith("weight_")]
    weight_df = bt.set_index("date")[weight_cols]
    weight_df.columns = [c.replace("weight_", "") for c in weight_df.columns]

    st.write("**Stock weights over time**")
    st.area_chart(weight_df)

    st.write("**Sentiment-tilted index value vs. equal-weight baseline** (normalized, start = 1.0)")
    value_df = bt.set_index("date")[["sentiment_tilted_value", "equal_weight_value"]]
    st.line_chart(value_df)

    final = bt.iloc[-1]
    delta_pct = (final["sentiment_tilted_value"] / final["equal_weight_value"] - 1) * 100
    v1, v2, v3 = st.columns(3)
    v1.metric("Final sentiment-tilted value", f"{final['sentiment_tilted_value']:.4f}")
    v2.metric("Final equal-weight value", f"{final['equal_weight_value']:.4f}")
    v3.metric("Delta vs. naive baseline", f"{delta_pct:+.2f}%")

with tab_b:
    st.subheader("Event-driven portfolio stress testing")
    st.write(
        "Synthetic wholesale-banking portfolio (loans, bonds, derivatives, equities). "
        "A stress test triggers whenever a signal's impact score exceeds the threshold."
    )
    threshold = st.slider("Impact score trigger threshold", min_value=1, max_value=10, value=7)

    portfolio = load_portfolio()
    shock_config = load_shock_config()
    triggers = find_trigger_events(signals, impact_threshold=threshold)

    st.write(f"**{len(triggers)} trigger event(s) found** at threshold > {threshold}")

    if len(triggers) == 0:
        st.info("No events cross this threshold. Lower the slider to see triggered stress tests.")
    else:
        options = [
            f"{row.date.date()} | {row.ticker} | {row.event_classification} (impact {row.impact_score})"
            for row in triggers.itertuples()
        ]
        selected = st.selectbox("Select a trigger event to inspect", options)
        selected_idx = options.index(selected)
        event = triggers.iloc[selected_idx]

        result = run_stress_test(event, portfolio, shock_config)

        st.write(f"> {result['event_text']}")

        r1, r2, r3 = st.columns(3)
        r1.metric("Portfolio value before", f"${result['pre_value']:,.0f}")
        r2.metric("Portfolio value after", f"${result['post_value']:,.0f}")
        r3.metric("Impact", f"{result['pct_change']:+.2f}%")

        breakdown = pd.DataFrame(result["breakdown_by_asset_type"]).set_index("asset_type")
        st.write("**Before vs. after, by asset class**")
        st.bar_chart(breakdown[["pre_value", "post_value"]])
        st.dataframe(breakdown, width="stretch")

        st.write("**All triggered events (most recent first)**")
        st.dataframe(triggers.sort_values("date", ascending=False), width="stretch")
