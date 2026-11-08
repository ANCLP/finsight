"""Discourse & Syntactic Clause Deconstructor for Financial News.

Provides linguistic pre-processing to eliminate the 5 critical failure modes of n-gram models:
1. Contrastive clause inversion ("Although X was great, Y was a complete disaster")
2. Directional metric pairing ("debt expanded" vs "revenue expanded")
3. Rumor/probe denial inversion ("Management denied reports of fraud")
4. Scare-quote attenuation ("CEO promises 'record' turnaround")
5. Consensus expectation comparison extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DeconstructedSentence:
    """Structured representation of a syntactically deconstructed sentence."""
    raw_text: str
    operative_clause: str
    concession_clause: Optional[str] = None
    contrastive_marker: Optional[str] = None
    is_contrastive: bool = False
    operative_weight: float = 1.0
    concession_weight: float = 0.0

    # Linguistic flags
    has_scare_quotes: bool = False
    is_rumor_denial: bool = False
    directional_metric_override: Optional[float] = None
    override_reason: Optional[str] = None


class DiscourseDeconstructor:
    """Deconstructs complex subordinate clauses and financial metric contexts."""

    # Contrastive markers that initiate a subordinate concession clause
    INITIAL_CONTRASTIVE_MARKERS = [
        r"^although\b",
        r"^even though\b",
        r"^despite\b",
        r"^in spite of\b",
        r"^while\b",
    ]

    # Contrastive markers that pivot mid-sentence to an operative main clause
    MID_CONTRASTIVE_MARKERS = [
        r",\s*(?:but|however|yet|nevertheless|nonetheless|whereas)\s+",
        r"\s+(?:but|however|yet)\s+",
    ]

    # Asymmetric financial metric nouns where increase = bearish and decrease = bullish
    LIABILITY_METRIC_NOUNS = {
        "debt", "debts", "obligation", "obligations", "debt obligations",
        "borrowing", "borrowings", "liability", "liabilities",
        "loss", "losses", "net loss", "deficit", "deficits", "expenses", "expense", "expenditure",
        "capex", "npa", "npas", "bad loans", "costs", "cost", "interest burden",
        "leverage", "indebtedness",
    }

    INCREASE_VERBS = {
        "expand", "expands", "expanded", "expanding", "expansion",
        "increase", "increases", "increased", "increasing",
        "rise", "rises", "rose", "risen", "rising",
        "surge", "surges", "surged", "surging",
        "jump", "jumps", "jumped", "jumping",
        "climb", "climbs", "climbed", "climbing",
        "widen", "widens", "widened", "widening",
        "balloon", "balloons", "ballooned", "ballooning",
        "grow", "grows", "grew", "grown", "growing",
        "spike", "spikes", "spiked", "spiking",
        "spiral", "spirals", "spiraled", "spiraling",
        "accumulate", "accumulates", "accumulated", "accumulating",
        "mount", "mounts", "mounted", "mounting",
        "higher", "soared", "soaring", "elevated",
    }

    DECREASE_VERBS = {
        "reduce", "reduces", "reduced", "reducing", "reduction",
        "fall", "falls", "fell", "fallen", "falling",
        "drop", "drops", "dropped", "dropping",
        "shrink", "shrinks", "shrank", "shrunk", "shrinking",
        "contract", "contracts", "contracted", "contracting",
        "decline", "declines", "declined", "declining",
        "cut", "cuts", "cutting",
        "slash", "slashes", "slashed", "slashing",
        "repay", "repays", "repaid", "repaying", "repayment",
        "prepay", "prepays", "prepaid", "prepaying", "prepayment",
        "clear", "clears", "cleared", "clearing",
        "settle", "settles", "settled", "settling",
        "curb", "curbs", "curbed", "curbing",
        "ease", "eases", "eased", "easing",
        "lower", "lowers", "lowered", "lowering",
    }

    # Scare-quote regex: captures single or double quoted superlative hype words
    SCARE_QUOTE_PATTERN = re.compile(
        r"['\"](?:record|revolutionary|milestone|stellar|historic|breakthrough|unprecedented|game-changer|turnaround)['\"]",
        re.IGNORECASE,
    )

    # Rumor denial patterns
    DENIAL_VERB_PATTERN = re.compile(
        r"\b(?:denied|denies|denying|rejects|rejected|rejecting|refutes|refuted|refuting|debunked|dismissed|dismisses|unfounded|baseless|no truth|categorically denies)\b",
        re.IGNORECASE,
    )
    ALARM_WORDS_PATTERN = re.compile(
        r"\b(?:probe|fraud|investigation|raid|searches|scandal|default|insolvency|charges|irregularities)\b",
        re.IGNORECASE,
    )

    @classmethod
    def deconstruct(cls, text: str) -> DeconstructedSentence:
        """Deconstructs a financial sentence into its operative and concession components."""
        cleaned = text.strip()
        lower = cleaned.lower()

        # 1. Check for scare quotes around superlative words
        has_scare_quotes = bool(cls.SCARE_QUOTE_PATTERN.search(cleaned))

        # 2. Check for rumor / probe denial
        is_rumor_denial = False
        denial_match = cls.DENIAL_VERB_PATTERN.search(lower)
        if denial_match:
            # Look for alarm words within the same sentence
            alarm_match = cls.ALARM_WORDS_PATTERN.search(lower)
            if alarm_match:
                is_rumor_denial = True

        # 3. Check for directional metric pairing (e.g. "debt expanded", "costs fell")
        metric_override, override_reason = cls._check_directional_metrics(lower)

        # 4. Check for initial contrastive markers ("Although [Clause A], [Clause B]")
        for pattern in cls.INITIAL_CONTRASTIVE_MARKERS:
            m = re.search(pattern, lower)
            if m:
                # Split at comma between clauses
                parts = cleaned.split(",", 1)
                if len(parts) == 2:
                    concession = parts[0].strip()
                    operative = parts[1].strip()
                    return DeconstructedSentence(
                        raw_text=cleaned,
                        operative_clause=operative,
                        concession_clause=concession,
                        contrastive_marker=m.group(0),
                        is_contrastive=True,
                        operative_weight=0.80,
                        concession_weight=0.20,
                        has_scare_quotes=has_scare_quotes,
                        is_rumor_denial=is_rumor_denial,
                        directional_metric_override=metric_override,
                        override_reason=override_reason,
                    )

        # 5. Check for mid-sentence contrastive markers ("[Clause A], but [Clause B]")
        for pattern in cls.MID_CONTRASTIVE_MARKERS:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                concession = cleaned[:match.start()].strip()
                operative = cleaned[match.end():].strip()
                return DeconstructedSentence(
                    raw_text=cleaned,
                    operative_clause=operative,
                    concession_clause=concession,
                    contrastive_marker=match.group(0).strip(" ,"),
                    is_contrastive=True,
                    operative_weight=0.80,
                    concession_weight=0.20,
                    has_scare_quotes=has_scare_quotes,
                    is_rumor_denial=is_rumor_denial,
                    directional_metric_override=metric_override,
                    override_reason=override_reason,
                )

        # Non-contrastive simple sentence
        return DeconstructedSentence(
            raw_text=cleaned,
            operative_clause=cleaned,
            concession_clause=None,
            contrastive_marker=None,
            is_contrastive=False,
            operative_weight=1.0,
            concession_weight=0.0,
            has_scare_quotes=has_scare_quotes,
            is_rumor_denial=is_rumor_denial,
            directional_metric_override=metric_override,
            override_reason=override_reason,
        )

    @classmethod
    def _check_directional_metrics(cls, lower_text: str) -> Tuple[Optional[float], Optional[str]]:
        """Identifies asymmetric metric contexts where normal bag-of-words verbs flip polarity."""
        tokens = re.findall(r"\b\w+\b", lower_text)

        for i, word in enumerate(tokens):
            if word in cls.LIABILITY_METRIC_NOUNS:
                # Look in forward and backward window of 4 tokens for growth or reduction verbs
                window_start = max(0, i - 4)
                window_end = min(len(tokens), i + 5)
                sub_window = tokens[window_start:window_end]

                for w in sub_window:
                    if w in cls.INCREASE_VERBS:
                        return -0.55, f"Liability metric '{word}' increased ('{w}') -> Bearish leverage"
                    if w in cls.DECREASE_VERBS:
                        return 0.50, f"Liability metric '{word}' decreased ('{w}') -> Bullish deleveraging"

        return None, None
