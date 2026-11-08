"""Financial text normalization, ticker preservation, and smart sentence boundary chunker.

Specialized for financial journalism where standard NLP tokenizers break on:
- Financial decimals (e.g., 'margin dropped to 2.4 % from 5.7 % .')
- Company suffixes (e.g., 'Apple Inc.', 'Tata Motors Ltd.', 'Elcoteq SE')
- Currencies and multipliers (e.g., 'EUR 20.5 mn', 'USD 1.2B', '₹500 Cr')
- Cashtags and tickers (e.g., '$AAPL', '$TCS', 'NSE:RELIANCE')
"""

from __future__ import annotations

import re
from typing import List

# Common abbreviations in financial texts that should NOT trigger a sentence split
ABBREVIATIONS = (
    r"(?:Inc|Ltd|Corp|Co|SE|Oyj|Plc|LLC|AG|SA|AB|GmbH|vs|e\.g|i\.e|vol|no|approx|est|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|U\.S|U\.K|E\.U|Govt|dept)"
)


class TextPreprocessor:
    """Preprocesses, cleans, and segments financial news and phrasebank texts."""

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Cleans and normalizes financial text while preserving financial numbers and tickers."""
        if not text:
            return ""

        # Normalize unicode quotes and dashes
        s = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        s = s.replace("–", "-").replace("—", "-").replace("…", "...")

        # Normalize spaces around percent signs and decimals common in PhraseBank:
        # e.g., '2.4 %' -> '2.4%' or 'EUR 20 mn' -> 'EUR 20mn'
        s = re.sub(r"(\d+)\s+%", r"\1%", s)

        # Remove HTML artifacts if any
        s = re.sub(r"<[^>]+>", " ", s)

        # Normalize excessive whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """Segments a multi-sentence financial article or paragraph into distinct sentences,

        respecting financial decimals, company abbreviations, and currency mentions.
        """
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []

        # Mask dots in numbers (e.g. 2.4 -> 2<DOT>4)
        protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", cleaned)

        # Mask dots in common financial abbreviations
        def mask_abbrev(match: re.Match[str]) -> str:
            return match.group(0).replace(".", "<DOT>")

        abbrev_pattern = re.compile(rf"\b{ABBREVIATIONS}\.", re.IGNORECASE)
        protected = abbrev_pattern.sub(mask_abbrev, protected)

        # Mask ellipsis
        protected = protected.replace("...", "<ELLIPSIS>")

        # Split on sentence boundaries: (. ! ?) followed by whitespace and capital letter / quote / number
        raw_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\$])", protected)

        sentences: List[str] = []
        for s in raw_splits:
            # Unmask dots
            restored = s.replace("<DOT>", ".").replace("<ELLIPSIS>", "...").strip()
            # If the sentence still has content
            if restored and len(restored) > 2:
                sentences.append(restored)

        # Fallback if no sentence split was detected
        return sentences if sentences else [cleaned]
