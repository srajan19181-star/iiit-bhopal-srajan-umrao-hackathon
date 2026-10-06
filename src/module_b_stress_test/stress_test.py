"""Module B: event-driven portfolio stress testing.

Walks the chronological risk-signal timeline; whenever a signal's
event_classification + impact_score crosses the trigger threshold, applies a
simplified shock (from data/raw/shock_config.json) to a synthetic
wholesale-banking portfolio. Each trigger is evaluated in isolation against
the ORIGINAL baseline portfolio (not cumulative/compounding across the
timeline) -- a deliberate simplification that avoids shock-ordering
ambiguity while still supporting multiple trigger events in the dashboard.
"""
import pandas as pd


def find_trigger_events(signals: pd.DataFrame, impact_threshold: int = 7,
                         categories: set | None = None) -> pd.DataFrame:
    """Rows of `signals` that qualify as a stress-test trigger, sorted by date."""
    triggered = signals[signals["impact_score"] > impact_threshold]
    if categories:
        triggered = triggered[triggered["event_classification"].isin(categories)]
    return triggered.sort_values("date").reset_index(drop=True)


def apply_shock(portfolio: pd.DataFrame, event_classification: str, shock_config: dict) -> pd.DataFrame:
    """Returns a copy of `portfolio` with shock_pct and shocked_value columns."""
    shocks = shock_config.get(event_classification, {})
    shocked = portfolio.copy()
    shocked["shock_pct"] = shocked["asset_type"].map(shocks).fillna(0.0)
    shocked["shocked_value"] = shocked["current_value"] * (1 + shocked["shock_pct"])
    return shocked


def run_stress_test(event: pd.Series, portfolio: pd.DataFrame, shock_config: dict) -> dict:
    """Single isolated scenario: shocks the ORIGINAL baseline portfolio.

    Returns pre/post totals, overall pct change, and a per-asset-class
    breakdown -- feeds the "before vs after" dashboard chart directly.
    """
    shocked = apply_shock(portfolio, event["event_classification"], shock_config)
    pre_value = float(portfolio["current_value"].sum())
    post_value = float(shocked["shocked_value"].sum())

    breakdown = (
        shocked.groupby("asset_type")
        .agg(pre_value=("current_value", "sum"), post_value=("shocked_value", "sum"))
        .reset_index()
    )
    breakdown["pct_change"] = (breakdown["post_value"] / breakdown["pre_value"] - 1) * 100

    return {
        "event_date": str(event["date"]),
        "event_classification": event["event_classification"],
        "event_company": event.get("company", ""),
        "event_text": event.get("text", ""),
        "impact_score": int(event["impact_score"]),
        "pre_value": pre_value,
        "post_value": post_value,
        "pct_change": (post_value / pre_value - 1) * 100,
        "breakdown_by_asset_type": breakdown.to_dict("records"),
    }
