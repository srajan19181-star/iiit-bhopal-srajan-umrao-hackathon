"""Module A: tactical, sentiment-driven index rebalancing.

Positive sentiment on a ticker nudges its weight up, negative sentiment
nudges it down; weights are renormalized to sum to 1 after every update.
`backtest` replays this day-by-day over the synthetic price series and
compares it against a static equal-weight baseline -- that comparison is
the "efficiency gains vs naive approach" evidence for the deck.
"""
import pandas as pd


def update_weights(weights: pd.Series, sentiment_scores: pd.Series,
                    k: float = 0.1, floor: float = 0.01, cap: float = 0.15) -> pd.Series:
    """new_w = clip(old_w * (1 + k*sentiment), floor, cap), then renormalized to sum=1.

    Tickers with no sentiment signal that day get sentiment=0 (no change).
    """
    sentiment = sentiment_scores.reindex(weights.index).fillna(0.0)
    adjusted = (weights * (1 + k * sentiment)).clip(lower=floor, upper=cap)
    return adjusted / adjusted.sum()


def backtest(prices: pd.DataFrame, signals: pd.DataFrame, index_config: pd.DataFrame,
             k: float = 0.1) -> pd.DataFrame:
    """Day-by-day replay over `prices` (long: ticker,date,price).

    Returns one row per day with date, sentiment_tilted_value, equal_weight_value
    (both normalized to start at 1.0), and a weight_<ticker> column per ticker --
    the weight columns are what the "weights over time" dashboard chart plots.
    """
    tickers = index_config["ticker"].tolist()
    price_pivot = prices.pivot(index="date", columns="ticker", values="price").sort_index()[tickers]
    dates = price_pivot.index.tolist()

    daily_sentiment = (
        signals.assign(date=pd.to_datetime(signals["date"]))
        .groupby(["date", "ticker"])["sentiment_score"].mean()
        .unstack("ticker")
        .reindex(index=price_pivot.index, columns=tickers)
    )

    weights = index_config.set_index("ticker")["start_weight"].reindex(tickers)
    equal_weights = pd.Series(1.0 / len(tickers), index=tickers)

    tilted_value, equal_value = 1.0, 1.0
    records = [{
        "date": dates[0], "sentiment_tilted_value": tilted_value, "equal_weight_value": equal_value,
        **{f"weight_{t}": weights[t] for t in tickers},
    }]

    for i in range(1, len(dates)):
        today, yesterday = dates[i], dates[i - 1]
        daily_return = price_pivot.loc[today] / price_pivot.loc[yesterday] - 1.0

        tilted_value *= 1 + (weights * daily_return).sum()
        equal_value *= 1 + (equal_weights * daily_return).sum()

        records.append({
            "date": today, "sentiment_tilted_value": tilted_value, "equal_weight_value": equal_value,
            **{f"weight_{t}": weights[t] for t in tickers},
        })

        weights = update_weights(weights, daily_sentiment.loc[today], k=k)

    return pd.DataFrame(records)
