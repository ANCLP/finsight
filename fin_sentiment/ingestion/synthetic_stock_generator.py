"""Synthetic financial stock news generator for modern market sectors.

Generates realistic, high-precision financial statements, earnings disclosures,
and catalyst headlines across US megacap tech, Indian bluechips, and global equities.
Designed to cure the 12.5% negative class deficit in FinancialPhraseBank and inject
modern financial vocabulary (AI capex, GPU demand, cloud ARR, regulatory notices, buybacks).
"""

from __future__ import annotations

import random
from typing import List, Optional

from fin_sentiment.types import DatasetItem, RawLabel

# Modern prominent stock symbols with sector context
MODERN_TICKERS = [
    # US Tech & Semis
    ("Apple", "$AAPL", "hardware and services"),
    ("Microsoft", "$MSFT", "cloud and enterprise software"),
    ("Nvidia", "$NVDA", "AI data center chips and accelerators"),
    ("Alphabet", "$GOOGL", "search advertising and Google Cloud"),
    ("Amazon", "$AMZN", "e-commerce and AWS cloud infrastructure"),
    ("Meta", "$META", "digital advertising and AI infrastructure"),
    ("Tesla", "$TSLA", "electric vehicles and energy storage"),
    ("Broadcom", "$AVGO", "networking semiconductors and custom ASICs"),
    ("AMD", "$AMD", "data center processors and graphics cards"),
    ("Qualcomm", "$QCOM", "mobile chipsets and automotive platforms"),
    # Indian Bluechips
    ("Tata Consultancy Services", "$TCS", "IT services and digital transformation"),
    ("Infosys", "$INFY", "enterprise IT consulting and digital cloud"),
    ("Reliance Industries", "$RELIANCE", "retail, telecom and oil-to-chemicals"),
    ("HDFC Bank", "$HDFCBANK", "retail banking and wholesale credit"),
    ("ICICI Bank", "$ICICIBANK", "private banking and corporate lending"),
    ("Tata Motors", "$TATAMOTORS", "commercial vehicles and luxury passenger cars"),
    ("State Bank of India", "$SBIN", "public sector banking and infrastructure lending"),
    ("Bharti Airtel", "$BHARTIARTL", "telecommunications and enterprise 5G services"),
    ("Larsen & Toubro", "$LT", "engineering, procurement and infrastructure construction"),
    ("Zomato", "$ZOMATO", "food delivery and quick-commerce dark stores"),
]

BULLISH_TEMPLATES = [
    "{company} ({ticker}) delivered strong quarterly earnings, posting a {pct}% YoY rise in net profit to exceed consensus analyst estimates.",
    "{company} reported quarterly revenue surged {pct}% to ${num} billion, beating Wall Street expectations on robust {sector} demand.",
    "Operating margins at {company} ({ticker}) expanded by {bps} basis points this quarter as operational efficiency improved.",
    "The board of directors of {company} approved a cash dividend of ${small_num} per share and authorized an expanded share buyback program.",
    "{company} ({ticker}) clinched a multi-year enterprise contract valued at over ${num}00 million for {sector}.",
    "{company} raised its full-year revenue and operating income guidance following record bookings in its {sector} division.",
    "Analysts at major investment banks upgraded {company} ({ticker}) to 'Overweight', citing accelerating free cash flow and margin expansion.",
    "{company} successfully completed its aggressive deleveraging program, reducing total debt obligations by {pct}%.",
    "Customer gross additions at {company} ({ticker}) reached an all-time high, driving double-digit revenue expansion.",
    "{company} reported record free cash flow of ${num}.{small_num} billion, up {pct}% compared to the prior-year period.",
    "Strong pricing power and supply chain optimization lifted gross margins for {company} ({ticker}) by {bps} bps.",
    "{company} announced that international revenue grew {pct}% YoY, driven by aggressive expansion in {sector}.",
]

BEARISH_TEMPLATES = [
    "{company} ({ticker}) missed consensus revenue estimates for the third quarter, citing discretionary spending slowdown in {sector}.",
    "{company} reported that net profit plunged {pct}% to ${num} million as cost inflation eroded operating margins.",
    "Operating margins at {company} ({ticker}) contracted by {bps} basis points due to severe price competition and higher input costs.",
    "{company} lowered its full-year earnings guidance and warned of prolonged headwinds across {sector}.",
    "Analysts downgraded {company} ({ticker}) to 'Sell', citing decelerating customer growth and mounting balance sheet debt.",
    "{company} announced plans to lay off {pct}% of its workforce as part of an emergency cost-reduction restructuring.",
    "Regulatory authorities issued a formal show-cause notice to {company} ({ticker}) following an investigation into accounting irregularities.",
    "Total borrowings and debt obligations at {company} ({ticker}) ballooned by {pct}% to ${num} billion, raising leverage concerns.",
    "Statutory auditor of {company} ({ticker}) resigned abruptly citing lack of adequate documentation on material transactions.",
    "{company} swung to an unexpected net loss of ${num} million, compared to a profit in the corresponding quarter last year.",
    "Customer churn at {company} ({ticker}) rose sharply while average revenue per user slipped {pct}% YoY.",
    "Shares of {company} ({ticker}) plunged {pct}% after management flagged substantial inventory write-downs in {sector}.",
]

NEUTRAL_TEMPLATES = [
    "{company} ({ticker}) announced it will release its quarterly financial results on {month} {day}.",
    "The board of directors of {company} scheduled its annual general meeting of shareholders for {month} {day}.",
    "{company} ({ticker}) published its annual sustainability and corporate governance report for the fiscal year.",
    "According to regulatory filings, {company} completed a scheduled refinancing of its existing credit facility.",
    "{company} ({ticker}) confirmed that senior leadership will participate in an upcoming industry investor conference.",
    "The company stated that its corporate headquarters will remain in its current location following operational review.",
    "{company} ({ticker}) submitted its periodic compliance disclosure in accordance with exchange listing requirements.",
    "Trading volume in shares of {company} ({ticker}) remained in line with the three-month historical daily average.",
    "{company} announced the routine appointment of a new independent non-executive director to its audit committee.",
    "The annual shareholder voting results for {company} ({ticker}) were submitted to regulatory authorities today.",
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


class SyntheticStockGenerator:
    """Generates synthetic financial news sentences with balanced Bull/Bear/Neutral classes."""

    @classmethod
    def generate_dataset(
        cls,
        n_samples: int = 1500,
        bull_ratio: float = 0.35,
        bear_ratio: float = 0.45,
        neutral_ratio: float = 0.20,
        random_state: Optional[int] = 42,
    ) -> List[DatasetItem]:
        """Generates a list of DatasetItem objects tailored to modern stock market news.

        Args:
            n_samples: Total number of samples to generate.
            bull_ratio: Proportion of bullish samples (default: 0.35).
            bear_ratio: Proportion of bearish samples to cure negative deficit (default: 0.45).
            neutral_ratio: Proportion of neutral corporate announcements (default: 0.20).
            random_state: Seed for deterministic reproducibility.

        Returns:
            List of generated DatasetItem objects.
        """
        rng = random.Random(random_state)
        items: List[DatasetItem] = []

        n_bull = int(n_samples * bull_ratio)
        n_bear = int(n_samples * bear_ratio)
        n_neut = n_samples - n_bull - n_bear

        # Generate Bullish samples
        for _ in range(n_bull):
            text = cls._render_sentence(BULLISH_TEMPLATES, rng)
            items.append(DatasetItem(
                text=text,
                sentiment=RawLabel.POSITIVE,
                agreement_level=1.0,
                source_file="synthetic_modern_stocks",
            ))

        # Generate Bearish samples
        for _ in range(n_bear):
            text = cls._render_sentence(BEARISH_TEMPLATES, rng)
            items.append(DatasetItem(
                text=text,
                sentiment=RawLabel.NEGATIVE,
                agreement_level=1.0,
                source_file="synthetic_modern_stocks",
            ))

        # Generate Neutral samples
        for _ in range(n_neut):
            text = cls._render_sentence(NEUTRAL_TEMPLATES, rng)
            items.append(DatasetItem(
                text=text,
                sentiment=RawLabel.NEUTRAL,
                agreement_level=1.0,
                source_file="synthetic_modern_stocks",
            ))

        rng.shuffle(items)
        return items

    @classmethod
    def _render_sentence(cls, templates: List[str], rng: random.Random) -> str:
        tpl = rng.choice(templates)
        company, ticker, sector = rng.choice(MODERN_TICKERS)
        pct = rng.randint(5, 65)
        bps = rng.randint(40, 320)
        num = rng.randint(2, 45)
        small_num = round(rng.uniform(0.15, 1.85), 2)
        month = rng.choice(MONTHS)
        day = rng.randint(1, 28)

        return tpl.format(
            company=company,
            ticker=ticker,
            sector=sector,
            pct=pct,
            bps=bps,
            num=num,
            small_num=small_num,
            month=month,
            day=day,
        )
