"""Generates all synthetic datasets under data/raw/.

Deterministic (seeded) so the repo's committed data/raw/* is reproducible by
re-running this script. All data is synthetic: no proprietary or real
confidential data is used. Real, publicly-known stock tickers are used for
realism in the mock index, but their price series are synthetic random walks
seeded from approximate public price levels as of early 2026 -- NOT pulled
from any live market data source.
"""
import csv
import json
import random
from pathlib import Path

RNG_SEED = 42
DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
N_DAYS = 60
START_DATE = "2026-08-01"

MOCK_INDEX = [
    ("AAPL", "Apple Inc.", "Technology", 230.0),
    ("MSFT", "Microsoft Corp.", "Technology", 420.0),
    ("GOOGL", "Alphabet Inc.", "Technology", 165.0),
    ("NVDA", "NVIDIA Corp.", "Technology", 135.0),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", 185.0),
    ("JPM", "JPMorgan Chase & Co.", "Financials", 210.0),
    ("BAC", "Bank of America Corp.", "Financials", 40.0),
    ("XOM", "Exxon Mobil Corp.", "Energy", 115.0),
    ("CVX", "Chevron Corp.", "Energy", 160.0),
    ("JNJ", "Johnson & Johnson", "Healthcare", 155.0),
    ("PFE", "Pfizer Inc.", "Healthcare", 28.0),
    ("PG", "Procter & Gamble Co.", "Consumer Staples", 165.0),
    ("KO", "Coca-Cola Co.", "Consumer Staples", 62.0),
    ("BA", "Boeing Co.", "Industrials", 180.0),
    ("CAT", "Caterpillar Inc.", "Industrials", 350.0),
]

EVENT_TEMPLATES = {
    "Geopolitical": {
        "positive": [
            "{company} rallies as easing geopolitical tensions revive trade talks in key export markets.",
            "{company} shares gain after a ceasefire agreement reduces regional supply-chain risk.",
        ],
        "negative": [
            "{company} shares slide as new sanctions target its overseas operations amid escalating tensions.",
            "{company} warns of disruption as conflict in a key sourcing region threatens its supply chain.",
            "{company} halts operations in affected region as geopolitical conflict intensifies.",
        ],
        "neutral": [
            "{company} issues a statement on its exposure to ongoing geopolitical developments.",
        ],
    },
    "Macroeconomic": {
        "positive": [
            "{company} rallies after stronger-than-expected GDP growth boosts consumer spending outlook.",
            "{company} shares climb as cooling inflation data raises hopes of interest rate cuts.",
        ],
        "negative": [
            "{company} warns of margin pressure as inflation and rising interest rates squeeze costs.",
            "{company} shares fall as weak jobs data stokes recession fears across the sector.",
        ],
        "neutral": [
            "{company} analysts await the next central bank rate decision before revising guidance.",
        ],
    },
    "Credit Event": {
        "positive": [
            "{company} credit outlook upgraded after a strong track record of debt repayment.",
            "{company} bonds rally as the firm announces early repayment of outstanding debt.",
        ],
        "negative": [
            "{company} bonds slump after a rating agency flags default risk amid mounting debt obligations.",
            "{company} shares drop on fears of a covenant breach following weak quarterly cash flow.",
            "{company} declares bankruptcy protection as default on senior debt becomes imminent.",
        ],
        "neutral": [
            "{company} refinances a portion of its debt load at prevailing market rates.",
        ],
    },
    "Merger/Acquisition": {
        "positive": [
            "{company} to acquire a smaller rival in a multi-billion dollar deal, shares jump on synergy hopes.",
            "{company} shares surge on takeover speculation from a larger industry player.",
        ],
        "negative": [
            "{company} merger talks collapse, deal uncertainty weighs heavily on the stock.",
            "{company} shares fall as regulators block its proposed acquisition on antitrust grounds.",
        ],
        "neutral": [
            "{company} confirms it is in early-stage discussions regarding a potential strategic deal.",
        ],
    },
    "Product Launch": {
        "positive": [
            "{company} unveils its next-generation product line, analysts praise the innovation.",
            "{company} shares pop after its newest product sells out within hours of launch.",
        ],
        "negative": [
            "{company} delays its flagship product launch amid ongoing supply chain issues.",
            "{company} shares dip after its new product launch draws a lukewarm reception.",
        ],
        "neutral": [
            "{company} schedules a product showcase event for investors and press next month.",
        ],
    },
    "Other": {
        "neutral": [
            "{company} is set to report quarterly earnings next week, in line with analyst expectations.",
            "{company} announces routine changes to its board of directors.",
            "{company} reiterates full-year guidance at an investor conference.",
        ],
        "positive": [
            "{company} beats quarterly earnings expectations, shares tick higher on the print.",
        ],
        "negative": [
            "{company} misses quarterly revenue expectations, shares tick lower on the print.",
        ],
    },
}

TWEET_TEMPLATES = {
    "Geopolitical": {
        "positive": ["{ticker} up on the ceasefire news, tensions finally cooling off #markets"],
        "negative": ["{ticker} getting hit hard, sanctions headlines are brutal rn", "war fears tanking {ticker} today, ugly tape"],
        "neutral": ["watching {ticker} react to the geopolitical headlines this morning"],
    },
    "Macroeconomic": {
        "positive": ["{ticker} ripping on the inflation print, rate cut hopes back on"],
        "negative": ["{ticker} bleeding, rates up again and nobody wants risk rn", "inflation data wrecking {ticker} premarket"],
        "neutral": ["{ticker} flat ahead of the Fed decision later today"],
    },
    "Credit Event": {
        "positive": ["{ticker} up after the ratings upgrade, debt load looking manageable now"],
        "negative": ["{ticker} default risk headlines are scary, bonds cratering", "{ticker} bankruptcy talk trending, not good"],
        "neutral": ["{ticker} refinancing news out, digesting the details"],
    },
    "Merger/Acquisition": {
        "positive": ["{ticker} 🚀 on the buyout rumor, deal premium incoming?"],
        "negative": ["{ticker} deal falling apart, merger arb traders not happy"],
        "neutral": ["{ticker} confirms talks, no deal terms yet"],
    },
    "Product Launch": {
        "positive": ["{ticker} new product looks insane, sold out in an hour lol"],
        "negative": ["{ticker} launch delayed again... supply chain still a mess"],
        "neutral": ["{ticker} teasing a launch event next month, curious what it is"],
    },
    "Other": {
        "neutral": ["{ticker} earnings next week, setting price alerts"],
        "positive": ["{ticker} crushed earnings, calls printing today"],
        "negative": ["{ticker} missed big, puts are free money rn"],
    },
}

SENTIMENT_VALUE = {"positive": 0.6, "negative": -0.6, "neutral": 0.0}

# Deterministic, guaranteed high-impact rows so Module B's trigger condition
# (impact_score > 7) never depends on random seed luck. Written directly,
# bypassing the template sampler.
GUARANTEED_HIGH_IMPACT = [
    # (days_ago, ticker, company, source, text, intended_sentiment, intended_event)
    (5, "XOM", "Exxon Mobil Corp.", "news",
     "Exxon Mobil Corp. shares plunge as new international sanctions and an escalating regional war "
     "threaten to shut down a major overseas production facility, raising fears of a prolonged supply crisis.",
     "negative", "Geopolitical"),
    (12, "JPM", "JPMorgan Chase & Co.", "news",
     "JPMorgan Chase & Co. bonds tumble after a major rating agency downgrades the firm, warning of imminent "
     "default risk tied to a sharply deteriorating debt position and bankruptcy concerns among key counterparties.",
     "negative", "Credit Event"),
    (20, "CAT", "Caterpillar Inc.", "news",
     "Caterpillar Inc. halts all operations in a conflict zone as war and new sanctions cut off its supply "
     "chain entirely, with analysts warning of a severe and prolonged earnings hit.",
     "negative", "Geopolitical"),
    (30, "BAC", "Bank of America Corp.", "twitter",
     "BAC in freefall, bankruptcy and default headlines everywhere, this is a crisis not a dip",
     "negative", "Credit Event"),
    (40, "CVX", "Chevron Corp.", "news",
     "Chevron Corp. declares a state of emergency as sanctions and open war in a core operating region force "
     "a full shutdown, with the company warning of default risk on regional debt obligations.",
     "negative", "Geopolitical"),
]


def _daterange(n_days: int):
    import datetime
    start = datetime.date.fromisoformat(START_DATE)
    return [start + datetime.timedelta(days=i) for i in range(n_days)]


def generate_text_dataset(rng: random.Random, source: str, templates: dict, n_rows: int, id_prefix: str):
    dates = _daterange(N_DAYS)
    rows = []
    categories = list(templates.keys())
    for i in range(n_rows):
        ticker, company, _sector, _price = rng.choice(MOCK_INDEX)
        category = rng.choice(categories)
        sentiment_bucket = rng.choice(list(templates[category].keys()))
        text_template = rng.choice(templates[category][sentiment_bucket])
        text = text_template.format(company=company, ticker=ticker)
        date = rng.choice(dates)
        rows.append({
            "id": f"{id_prefix}_{i:04d}",
            "date": date.isoformat(),
            "source": source,
            "company": company,
            "ticker": ticker,
            "text": text,
            "intended_sentiment": sentiment_bucket,
            "intended_event": category,
        })
    return rows


def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_news_and_tweets():
    rng = random.Random(RNG_SEED)
    news_rows = generate_text_dataset(rng, "news", EVENT_TEMPLATES, 350, "news")
    tweet_rows = generate_text_dataset(rng, "twitter", TWEET_TEMPLATES, 500, "tw")

    import datetime
    start = datetime.date.fromisoformat(START_DATE)
    for idx, (days_ago, ticker, company, source, text, sentiment, event) in enumerate(GUARANTEED_HIGH_IMPACT):
        row = {
            "id": f"hi_{idx:02d}",
            "date": (start + datetime.timedelta(days=days_ago)).isoformat(),
            "source": source,
            "company": company,
            "ticker": ticker,
            "text": text,
            "intended_sentiment": sentiment,
            "intended_event": event,
        }
        if source == "news":
            news_rows.append(row)
        else:
            tweet_rows.append(row)

    news_rows.sort(key=lambda r: r["date"])
    tweet_rows.sort(key=lambda r: r["date"])

    fieldnames = ["id", "date", "source", "company", "ticker", "text", "intended_sentiment", "intended_event"]
    write_csv(DATA_RAW / "news_headlines.csv", news_rows, fieldnames)
    write_csv(DATA_RAW / "tweets.csv", tweet_rows, fieldnames)
    print(f"Wrote {len(news_rows)} news rows, {len(tweet_rows)} tweet rows "
          f"(including {len(GUARANTEED_HIGH_IMPACT)} guaranteed high-impact rows).")
    return news_rows, tweet_rows


def generate_mock_index():
    n = len(MOCK_INDEX)
    rows = [
        {"ticker": t, "company": c, "sector": s, "start_weight": round(1.0 / n, 6)}
        for t, c, s, _p in MOCK_INDEX
    ]
    write_csv(DATA_RAW / "mock_index.csv", rows, ["ticker", "company", "sector", "start_weight"])
    print(f"Wrote mock_index.csv with {n} tickers.")


def _build_sentiment_map(news_rows: list, tweet_rows: list) -> dict:
    """(ticker, date_iso) -> mean numeric intended_sentiment for that ticker/day."""
    buckets: dict = {}
    for row in news_rows + tweet_rows:
        key = (row["ticker"], row["date"])
        buckets.setdefault(key, []).append(SENTIMENT_VALUE[row["intended_sentiment"]])
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def generate_mock_index_prices(sentiment_map: dict):
    """Synthetic daily price walk per ticker.

    NOTE on a deliberate stylized assumption: each day's drift is nudged by a
    decayed running average of that ticker's *intended* sentiment (the label
    baked into the text generator, not the engine's measured output). This
    gives Module A's sentiment-tilt mechanism something real to demonstrate
    against the equal-weight baseline. Real-world sentiment/forward-return
    correlation is far weaker and noisier than modeled here -- this is a
    hackathon demo convenience, not a claim about real markets, and is
    called out explicitly in the README's dataset-assumptions section.
    """
    rng = random.Random(RNG_SEED + 1)
    dates = _daterange(N_DAYS)
    effect_strength = 0.006
    decay = 0.85
    rows = []
    for ticker, _company, _sector, start_price in MOCK_INDEX:
        price = start_price
        running_sentiment = 0.0
        for d in dates:
            day_value = sentiment_map.get((ticker, d.isoformat()))
            running_sentiment = decay * running_sentiment + (day_value if day_value is not None else 0.0)
            daily_return = rng.gauss(0.0003, 0.018) + effect_strength * running_sentiment
            price = max(1.0, price * (1 + daily_return))
            rows.append({"ticker": ticker, "date": d.isoformat(), "price": round(price, 2)})
    write_csv(DATA_RAW / "mock_index_prices.csv", rows, ["ticker", "date", "price"])
    print(f"Wrote mock_index_prices.csv with {len(rows)} rows ({N_DAYS} days x {len(MOCK_INDEX)} tickers).")


ASSET_TYPES = ["loan", "bond", "derivative", "equity"]
SECTORS = ["Technology", "Financials", "Energy", "Healthcare", "Industrials", "Consumer Staples"]


def generate_synthetic_portfolio():
    rng = random.Random(RNG_SEED + 2)
    rows = []
    n_assets = 40
    for i in range(n_assets):
        asset_type = ASSET_TYPES[i % len(ASSET_TYPES)]
        sector = rng.choice(SECTORS)
        notional = round(rng.uniform(2_000_000, 50_000_000), 2)
        rate = round(rng.uniform(0.02, 0.08), 4)
        maturity_years = rng.choice([1, 2, 3, 5, 7, 10])
        current_value = round(notional * rng.uniform(0.97, 1.03), 2)
        rows.append({
            "asset_id": f"A{i:03d}",
            "asset_type": asset_type,
            "sector": sector,
            "notional": notional,
            "rate": rate,
            "maturity_years": maturity_years,
            "current_value": current_value,
        })
    write_csv(DATA_RAW / "synthetic_portfolio.csv", rows,
              ["asset_id", "asset_type", "sector", "notional", "rate", "maturity_years", "current_value"])
    print(f"Wrote synthetic_portfolio.csv with {n_assets} assets across {ASSET_TYPES}.")


def generate_shock_config():
    shocks = {
        "Geopolitical": {"equity": -0.10, "derivative": -0.15, "bond": -0.04, "loan": -0.02},
        "Macroeconomic": {"equity": -0.05, "derivative": -0.08, "bond": -0.06, "loan": -0.03},
        "Credit Event": {"equity": -0.03, "derivative": -0.10, "bond": -0.12, "loan": -0.20},
        "Merger/Acquisition": {"equity": 0.04, "derivative": 0.02, "bond": 0.0, "loan": 0.0},
        "Product Launch": {"equity": 0.03, "derivative": 0.01, "bond": 0.0, "loan": 0.0},
        "Other": {"equity": 0.0, "derivative": 0.0, "bond": 0.0, "loan": 0.0},
    }
    path = DATA_RAW / "shock_config.json"
    path.write_text(json.dumps(shocks, indent=2), encoding="utf-8")
    print("Wrote shock_config.json.")


if __name__ == "__main__":
    news_rows, tweet_rows = generate_news_and_tweets()
    generate_mock_index()
    sentiment_map = _build_sentiment_map(news_rows, tweet_rows)
    generate_mock_index_prices(sentiment_map)
    generate_synthetic_portfolio()
    generate_shock_config()
    print("Done. All synthetic data written to", DATA_RAW)
