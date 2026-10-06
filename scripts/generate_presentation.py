"""Generates docs/presentation.pdf -- a 7-slide deck, pure Python (fpdf2).

Run AFTER generate_result_charts.py so the chart PNGs exist:
    python scripts/generate_presentation.py
"""
import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
DATA_OUTPUT = ROOT / "data" / "output"

PAGE_W, PAGE_H = 297, 167  # 16:9-ish landscape, mm
MARGIN = 14
NAVY = (31, 42, 68)
ACCENT = (58, 110, 165)
GREY = (90, 90, 90)


class Slide(FPDF):
    def header(self):
        pass

    def footer(self):
        pass


def new_slide(pdf: Slide):
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, PAGE_W, PAGE_H, "F")


def title_bar(pdf: Slide, kicker: str, title: str):
    pdf.set_xy(MARGIN, 10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 6, kicker.upper())
    pdf.set_xy(MARGIN, 16)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 12, title)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.8)
    pdf.line(MARGIN, 30, PAGE_W - MARGIN, 30)


def bullets(pdf: Slide, items: list, x: float, y: float, w: float, font_size: int = 13, gap: float = 9.5):
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_text_color(40, 40, 40)
    cy = y
    for item in items:
        pdf.set_xy(x, cy)
        pdf.set_font("Helvetica", "B", font_size)
        pdf.set_text_color(*ACCENT)
        pdf.cell(5, gap, "-")
        pdf.set_xy(x + 5, cy)
        pdf.set_font("Helvetica", "", font_size)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(w - 5, gap * 0.62, item)
        cy = pdf.get_y() + 2
    return cy


def footer_candidate(pdf: Slide, page_num: int):
    pdf.set_xy(MARGIN, PAGE_H - 10)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, "Srajan Umrao | IIIT Bhopal | S&P Global & Crisil Campus Hackathon 2026")
    pdf.set_xy(PAGE_W - MARGIN - 15, PAGE_H - 10)
    pdf.cell(15, 5, f"{page_num}/7", align="R")


def load_summary() -> dict:
    path = DATA_OUTPUT / "engine_eval_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main():
    summary = load_summary()
    pdf = Slide(orientation="L", unit="mm", format=(PAGE_W, PAGE_H))
    pdf.set_auto_page_break(False)

    # Slide 1: Title
    new_slide(pdf)
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, PAGE_W, PAGE_H, "F")
    pdf.set_xy(0, 55)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(PAGE_W, 16, "AI/NLP Financial Risk Engine", align="C")
    pdf.set_xy(0, 74)
    pdf.set_font("Helvetica", "", 15)
    pdf.set_text_color(200, 210, 225)
    pdf.cell(PAGE_W, 10, "Real-time unstructured data to actionable financial risk signals", align="C")
    pdf.set_xy(0, 100)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(PAGE_W, 8, "Srajan Umrao", align="C")
    pdf.set_xy(0, 108)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(190, 200, 215)
    pdf.cell(PAGE_W, 7, "IIIT Bhopal  |  S&P Global & Crisil Campus Hackathon 2026", align="C")

    # Slide 2: Problem & Approach
    new_slide(pdf)
    title_bar(pdf, "Slide 2", "Problem & Approach")
    bullets(pdf, [
        "Markets react to unstructured text (news, social media) faster than analysts can read it manually.",
        "Goal: a unified AI/NLP Risk Engine that converts raw financial text into structured, machine-readable "
        "risk signals -- sentiment, event category, and a severity (impact) score.",
        "Approach: ingest from two independent text sources (synthetic news headlines + tweets), run a hybrid "
        "NLP pipeline (pretrained transformers with an automatic rule-based fallback), then feed the structured "
        "signals into two downstream applications that demonstrate real trading and risk-management use cases.",
    ], MARGIN, 40, PAGE_W - 2 * MARGIN)
    footer_candidate(pdf, 2)

    # Slide 3: System Design
    new_slide(pdf)
    title_bar(pdf, "Slide 3", "System Design")
    arch_path = DOCS / "architecture.png"
    if arch_path.exists():
        pdf.image(str(arch_path), x=PAGE_W / 2 - 55, y=34, w=110)
    footer_candidate(pdf, 3)

    # Slide 4: Implementation Highlights
    new_slide(pdf)
    title_bar(pdf, "Slide 4", "Implementation Highlights")
    bullets(pdf, [
        "Hybrid NLP: FinBERT (sentiment) + a DeBERTa-v3-xsmall zero-shot classifier (event category), with an "
        "automatic VADER + keyword-rule fallback if the transformer stack is unavailable -- zero hard dependency.",
        "Impact scoring is a transparent, explainable heuristic (category severity x sentiment magnitude x "
        "classifier confidence x keyword hits), not a black box -- a deliberate, documented simplification.",
        "Module A (Tactical Rebalancer): vectorized pandas weight-update rule, backtested against a static "
        "equal-weight baseline for a direct 'efficiency vs. naive approach' comparison.",
        "Module B (Stress Testing): synthetic wholesale-banking portfolio (loans/bonds/derivatives/equities); "
        "isolated per-event shocks driven by a configurable event-to-shock mapping.",
        "Streamlit dashboard unifies both modules; signals are also served via a FastAPI endpoint and committed "
        "CSV/JSON files for downstream consumption.",
    ], MARGIN, 40, PAGE_W - 2 * MARGIN, font_size=11.5, gap=8.2)
    footer_candidate(pdf, 4)

    # Slide 5: Key Results
    new_slide(pdf)
    title_bar(pdf, "Slide 5", "Key Results")
    chart_path = ASSETS / "weight_drift.png"
    if chart_path.exists():
        pdf.image(str(chart_path), x=MARGIN, y=36, w=PAGE_W - 2 * MARGIN)
    else:
        n = summary.get("n_total_signals", "N/A")
        hi = summary.get("n_high_impact_signals_gt7", "N/A")
        bullets(pdf, [
            f"{n} risk signals generated from synthetic news + tweets.",
            f"{hi} high-impact (>7) signals correctly triggered Module B stress tests.",
        ], MARGIN, 40, PAGE_W - 2 * MARGIN)
    footer_candidate(pdf, 5)

    # Slide 6: Domain Impact
    new_slide(pdf)
    title_bar(pdf, "Slide 6", "Domain Impact")
    bullets(pdf, [
        "Risk and ratings analytics teams (the core of S&P Global & Crisil's business) need to react to breaking "
        "events faster than manual review allows -- this engine compresses that reaction time to seconds.",
        "Module A shows how sentiment intelligence can tactically tilt a portfolio toward resilience, without "
        "replacing a fundamentals-driven index methodology.",
        "Module B gives wholesale banking risk teams a fast, event-triggered way to gauge portfolio exposure to "
        "geopolitical and credit shocks -- directly relevant to Crisil's stress-testing and ratings workflows.",
        "The structured signal schema (sentiment, event, impact) is a reusable contract that any future "
        "downstream tool -- not just these two modules -- can subscribe to.",
    ], MARGIN, 40, PAGE_W - 2 * MARGIN)
    footer_candidate(pdf, 6)

    # Slide 7: Limitations & Next Steps
    new_slide(pdf)
    title_bar(pdf, "Slide 7", "Limitations & Next Steps")
    bullets(pdf, [
        "All data is synthetic and self-generated; self-consistency metrics check the engine against our own "
        "generator's intended labels, not an independently human-labeled benchmark.",
        "Impact scoring and stress shocks are simplified, explainable heuristics, not calibrated risk models.",
        "Module B treats each trigger event in isolation (not cumulative across the timeline) to avoid "
        "shock-ordering ambiguity -- a reasonable but simplifying assumption.",
        "Next steps: swap in live sources (GDELT, a news API) behind the same ingestion interface; fine-tune "
        "the event classifier on labeled financial-event data; add duration-based rate-shock modeling to Module B.",
    ], MARGIN, 40, PAGE_W - 2 * MARGIN)
    footer_candidate(pdf, 7)

    DOCS.mkdir(parents=True, exist_ok=True)
    out_path = DOCS / "presentation.pdf"
    pdf.output(str(out_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
