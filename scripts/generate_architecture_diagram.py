"""Generates docs/architecture.png -- a box-and-arrow system diagram.

Pure matplotlib (no Graphviz/system binary dependency) so it's portable.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).resolve().parent.parent / "docs"

COLORS = {
    "source": "#CDE7F0",
    "engine": "#FFE5B4",
    "signal": "#D9EAD3",
    "module": "#E6D7F5",
    "dash": "#F7D6D0",
    "border": "#4A4A4A",
    "text": "#1A1A1A",
}


def box(ax, xy, w, h, text, color, fontsize=10.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=COLORS["border"], facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=COLORS["text"], wrap=True, linespacing=1.4)
    return (x + w / 2, y, x + w / 2, y + h)  # (bottom_x, bottom_y, top_x, top_y)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.3, color=COLORS["border"], shrinkA=2, shrinkB=2,
    ))


def main():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("AI/NLP Financial Risk Engine - Architecture", fontsize=15, fontweight="bold", pad=14)

    # Row 1: sources
    news = box(ax, (0.6, 7.6), 3.0, 0.9, "Synthetic News\nHeadlines (CSV)", COLORS["source"])
    tweets = box(ax, (4.1, 7.6), 3.0, 0.9, "Synthetic Tweets\n(CSV)", COLORS["source"])

    # Row 2: ingestion
    ingest = box(ax, (1.6, 6.3), 5.0, 0.8, "Ingestion - unify_sources()\n(common schema)", COLORS["engine"])

    # Row 3: hybrid NLP
    nlp = box(ax, (0.6, 4.7), 7.0, 1.3,
              "Hybrid NLP Engine\nFinBERT (sentiment)  +  DeBERTa-v3-xsmall zero-shot (event)\n"
              "auto-fallback -> VADER + keyword rules",
              COLORS["engine"], fontsize=10)

    # Row 4: impact scoring
    impact = box(ax, (1.6, 3.5), 5.0, 0.8, "Impact Scoring Heuristic\n(severity x sentiment x confidence x keywords)",
                 COLORS["engine"], fontsize=9.5)

    # Row 5: signals
    signals = box(ax, (1.6, 2.2), 5.0, 0.9,
                  "Structured Risk Signals\nCSV / JSON file  +  FastAPI (/analyze, /signals)",
                  COLORS["signal"], fontsize=10)

    # Row 6: modules
    mod_a = box(ax, (0.3, 0.9), 3.4, 0.9, "Module A\nTactical Index Rebalancer", COLORS["module"])
    mod_b = box(ax, (4.3, 0.9), 3.4, 0.9, "Module B\nPortfolio Stress Testing", COLORS["module"])

    # Row 7: dashboard
    ax.set_ylim(-1.1, 9)
    dashboard = box(ax, (1.0, -1.0), 6.2, 0.8, "Streamlit Dashboard (3 tabs: Overview / Module A / Module B)",
                     COLORS["dash"], fontsize=10)

    # Arrows
    arrow(ax, (news[2], news[1]), (ingest[2] - 1.0, ingest[3]))
    arrow(ax, (tweets[2], tweets[1]), (ingest[2] + 1.0, ingest[3]))
    arrow(ax, (ingest[2], ingest[1]), (nlp[2], nlp[3]))
    arrow(ax, (nlp[2], nlp[1]), (impact[2], impact[3]))
    arrow(ax, (impact[2], impact[1]), (signals[2], signals[3]))
    arrow(ax, (signals[2] - 1.3, signals[1]), (mod_a[2], mod_a[3]))
    arrow(ax, (signals[2] + 1.3, signals[1]), (mod_b[2], mod_b[3]))
    arrow(ax, (mod_a[2], mod_a[1]), (dashboard[2] - 1.3, dashboard[3]))
    arrow(ax, (mod_b[2], mod_b[1]), (dashboard[2] + 1.3, dashboard[3]))

    DOCS.mkdir(parents=True, exist_ok=True)
    out_path = DOCS / "architecture.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
