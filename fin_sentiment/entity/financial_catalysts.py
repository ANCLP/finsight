"""Financial catalyst and corporate event detection engine.

Identifies market-moving catalysts (earnings surprises, margin changes, dividend declarations,
debt restructuring, management transitions, regulatory penalties) and assesses their directional bias.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from fin_sentiment.types import CatalystMatch, CatalystType, SentimentLabel

# Definition of catalysts with regex patterns, sentiment bias, and market impact weight
CATALYST_DEFINITIONS: List[Tuple[CatalystType, str, SentimentLabel, float]] = [
    # Bullish Catalysts
    (
        CatalystType.EARNINGS_BEAT,
        r"\b(?:beats?|topped|exceeded|outperformed)\s+(?:(?:revenue|earnings|profit|sales|quarterly)\s+)?(?:estimates?|forecasts?|consensus|expectations?)\b|"
        r"\b(?:profit|net profit|pat|operating profit)\s+(?:rose|surged|jumped|doubled|tripled|increased|grew)\b",
        SentimentLabel.BULLISH,
        0.18,
    ),
    (
        CatalystType.REVENUE_GROWTH,
        r"\b(?:revenue|sales|turnover|net sales)\s+(?:surged|rose|jumped|increased|climbed|up by|grew by)\b|"
        r"\bdouble-digit\s+(?:growth|sales growth|revenue growth)\b",
        SentimentLabel.BULLISH,
        0.14,
    ),
    (
        CatalystType.MARGIN_EXPANSION,
        r"\b(?:(?:refining|gross|ebitda|operating)?\s*margins?)\s+(?:expanded|widened|improved|rose|jumped)\b|"
        r"\bexpansion in margins?\b",
        SentimentLabel.BULLISH,
        0.14,
    ),
    (
        CatalystType.DIVIDEND_BUYBACK,
        r"\b(?:declared|approved|announced|recommends?)\s+(?:dividend|interim dividend|special dividend|share buyback|share repurchase)\b",
        SentimentLabel.BULLISH,
        0.12,
    ),
    (
        CatalystType.CONTRACT_WIN,
        r"\b(?:bags?|wins?|secured|awarded|clinched)\s+(?:order|contract|deal|mandate|project)\b|"
        r"\bmulti-million\s+(?:dollar|euro|rupee)\s+(?:deal|contract)\b",
        SentimentLabel.BULLISH,
        0.14,
    ),
    (
        CatalystType.RATING_UPGRADE,
        r"\b(?:upgraded|raises target|boosted target|target raised|initiates with buy|reiterates buy|outperform rating)\b",
        SentimentLabel.BULLISH,
        0.15,
    ),
    (
        CatalystType.DEBT_REDUCTION,
        r"\b(?:debt-free|prepaid debt|deleverage|deleveraging)\b|"
        r"\b(?:debt|borrowings?|liabilities|obligations?)\s+(?:reduced|declined|dropped|fell|halved|contracted)\b|"
        r"\b(?:reduces?|reducing|reduced|cuts?|cutting|slashed|paid off|paying off)\s+(?:(?:its|more|further)\s+)?(?:debt|borrowings?|liabilities|obligations?)\b",
        SentimentLabel.BULLISH,
        0.14,
    ),
    (
        CatalystType.PRICE_BREAKOUT,
        r"\b(?:jumps? past|surges? past|rall(?:y|ies|ied)|breakout|record high|all-time high|best day in|upside|surges?|gains?)\b",
        SentimentLabel.BULLISH,
        0.15,
    ),

    # Bearish Catalysts
    (
        CatalystType.EARNINGS_MISS,
        r"\b(?:misses?|missed|lagged|trailed|fell short of)\s+(?:(?:revenue|earnings|profit|sales|quarterly)\s+)?(?:estimates?|forecasts?|consensus|expectations?)\b|"
        r"\b(?:net loss|widened loss|loss of|swung to loss|slumped into loss)\b",
        SentimentLabel.BEARISH,
        0.18,
    ),
    (
        CatalystType.REVENUE_DECLINE,
        r"\b(?:revenue|sales|net sales|turnover)\s+(?:declined|dropped|fell|slumped|plunged|sank|down by)\b",
        SentimentLabel.BEARISH,
        0.14,
    ),
    (
        CatalystType.MARGIN_CONTRACTION,
        r"\b(?:(?:refining|gross|ebitda|operating)?\s*margins?)\s+(?:contracted|shrank|compressed|declined|came down to|eroded)\b|"
        r"\bmargin pressure\b",
        SentimentLabel.BEARISH,
        0.14,
    ),
    (
        CatalystType.LAYOFF_RESTRUCTURING,
        r"\b(?:laid off|lay off|laying off|layoffs|cutting|slashing|job cuts|reducing headcount|contracted the ranks|firing)\b",
        SentimentLabel.BEARISH,
        0.14,
    ),
    (
        CatalystType.REGULATORY_FINE,
        r"\b(?:fined|penalty|probe|investigation|fraud|subpoena|regulatory scrutiny|sebi order|sec charges|litigation)\b",
        SentimentLabel.BEARISH,
        0.18,
    ),
    (
        CatalystType.AUDITOR_RESIGNATION,
        r"\b(?:auditors?|statutory auditor)\s+(?:resigns?|quit|stepped down|refuses to sign|red-flagged)\b|\bresignation of (?:the )?(?:statutory )?auditor\b",
        SentimentLabel.BEARISH,
        0.20,
    ),
    (
        CatalystType.SEC_SEBI_PROBE,
        r"\b(?:sebi|sec|doj|ed|cbi)\s+(?:order|charges|scrutiny|probe|investigation|notice|summons?|raid|searches)\b|\bshow-cause notice\b|\bforensic audit\b",
        SentimentLabel.BEARISH,
        0.18,
    ),
    (
        CatalystType.FILING_DELAY,
        r"\b(?:delay(?:ed|s)?|fails? to (?:file|submit)|misses deadline for)\s+(?:annual report|quarterly results?|10-k|financial results?|audit)\b",
        SentimentLabel.BEARISH,
        0.16,
    ),
    (
        CatalystType.COUPON_DEFAULT,
        r"\b(?:defaults?|defaulted on|misses? (?:coupon|interest|principal|bond) payment|insolvency|bankruptcy|nclt)\b",
        SentimentLabel.BEARISH,
        0.20,
    ),
    (
        CatalystType.DEBT_EXPANSION,
        r"\b(?:debt|borrowings?|liabilities|obligations?)\s+(?:surged|rose|expanded|jumped|increased|ballooned)\b|"
        r"\b(?:expands?|expanding|expanded|increases?|increasing|increased|raising|raises?|taking on|takes on)\s+(?:(?:its|more|further)\s+)?(?:debt|borrowings?|liabilities|obligations?)\b",
        SentimentLabel.BEARISH,
        0.14,
    ),
    (
        CatalystType.RATING_DOWNGRADE,
        r"\b(?:downgraded|cuts target|slashed target|target reduced|initiates with sell|underperform rating|downgrades to sell)\b",
        SentimentLabel.BEARISH,
        0.15,
    ),
    (
        CatalystType.PRICE_BREAKDOWN,
        r"\b(?:slumps?|plunges?|dives?|tanks?|drags? lower|worst day in|52-week low|cracks?|steep fall|drops? sharply)\b",
        SentimentLabel.BEARISH,
        0.15,
    ),
]


class CatalystExtractor:
    """Extracts and evaluates directional financial catalysts from news sentences."""

    @classmethod
    def detect_catalysts(cls, text: str) -> List[CatalystMatch]:
        """Scans text for known financial market catalysts and returns matches."""
        if not text:
            return []

        matches: List[CatalystMatch] = []
        for cat_type, pattern, bias, weight in CATALYST_DEFINITIONS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                matches.append(
                    CatalystMatch(
                        catalyst=cat_type,
                        matched_phrase=m.group(0),
                        sentiment_bias=bias,
                        weight=weight,
                    )
                )
        return matches

    @classmethod
    def compute_catalyst_boost(cls, catalysts: List[CatalystMatch]) -> float:
        """Calculates a normalized directional score delta (-0.20 to +0.20) based on detected catalysts."""
        if not catalysts:
            return 0.0

        net_score = 0.0
        for c in catalysts:
            if c.sentiment_bias == SentimentLabel.BULLISH:
                net_score += c.weight
            elif c.sentiment_bias == SentimentLabel.BEARISH:
                net_score -= c.weight

        # Normalize catalyst influence within a bounded directional range
        return max(-0.30, min(0.30, round(net_score, 4)))
