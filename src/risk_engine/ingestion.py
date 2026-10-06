"""Loaders that unify the news and twitter sources into one schema."""
from pathlib import Path

import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

UNIFIED_COLUMNS = ["id", "date", "source", "company", "ticker", "text", "intended_sentiment", "intended_event"]


def load_news(path: Path = DATA_RAW / "news_headlines.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_tweets(path: Path = DATA_RAW / "tweets.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def unify_sources(news: pd.DataFrame | None = None, tweets: pd.DataFrame | None = None) -> pd.DataFrame:
    news = news if news is not None else load_news()
    tweets = tweets if tweets is not None else load_tweets()
    combined = pd.concat([news, tweets], ignore_index=True)[UNIFIED_COLUMNS]
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined
