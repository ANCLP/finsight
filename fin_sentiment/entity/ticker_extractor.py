"""Stock ticker and company entity detection engine.

Detects:
- Cashtags: $AAPL, $TCS, $NVDA, $RELIANCE
- Exchange prefixed tickers: NSE:TCS, BSE:500325, NASDAQ:MSFT, NYSE:BABA
- Suffix tickers: RELIANCE.NS, TCS.BO, AAPL.O
- Known prominent global & Indian corporate names (Tata Motors, Reliance Industries, Apple, Microsoft, Nvidia, Infosys, etc.)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

# Common prominent stock symbols and company aliases mapping to canonical ticker
COMMON_COMPANY_TICKERS: Dict[str, str] = {
    # Indian Bluechips & High-Volume Equities
    "reliance industries": "RELIANCE",
    "reliance": "RELIANCE",
    "tata consultancy services": "TCS",
    "tcs": "TCS",
    "infosys": "INFY",
    "hdfc bank": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "state bank of india": "SBIN",
    "sbi": "SBIN",
    "tata motors": "TATAMOTORS",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "larsen & toubro": "LT",
    "l&t": "LT",
    "itc": "ITC",
    "wipro": "WIPRO",
    "hcl tech": "HCLTECH",
    "hcl technologies": "HCLTECH",
    "bajaj finance": "BAJFINANCE",
    "maruti suzuki": "MARUTI",
    "maruti": "MARUTI",
    "titan company": "TITAN",
    "titan": "TITAN",
    "adani enterprises": "ADANIENT",
    "adani ports": "ADANIPORTS",
    "sun pharma": "SUNPHARMA",
    "persistent systems": "PERSISTENT",
    "persistent": "PERSISTENT",

    # US / Global Tech & Bluechips
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "broadcom": "AVGO",
    "jpmorgan chase": "JPM",
    "jpmorgan": "JPM",
    "berkshire hathaway": "BRK",
    "taiwan semiconductor": "TSM",
    "tsmc": "TSM",
    "amd": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
}

# Words that look like tickers but are common English words
TICKER_STOPWORDS = {
    "A", "I", "AN", "AM", "AT", "BY", "FOR", "IN", "IS", "IT", "OF", "ON", "OR", "TO",
    "UP", "US", "UK", "EU", "THE", "AND", "BUT", "ALL", "NEW", "NOW", "ONE", "TWO",
    "CEO", "CFO", "CTO", "COO", "IPO", "GDP", "CPI", "PE", "EPS", "ROE", "ROCE", "NAV",
    "YOY", "QOQ", "Q1", "Q2", "Q3", "Q4", "FY", "SEC", "RBI", "SEBI", "FED", "EST",
    "BUY", "SELL", "HOLD", "GAIN", "LOSS", "BEAT", "MISS", "HIGH", "LOW", "DEBT", "BANK",
}


class TickerExtractor:
    """Extracts stock tickers and entity mentions from financial text."""

    # Regex for cashtags ($AAPL, $RELIANCE)
    CASHTAG_RE = re.compile(r"\$([A-Za-z0-9\.\_]{1,12})\b")

    # Regex for exchange prefixed tickers (NSE:TCS, NASDAQ:AAPL)
    EXCHANGE_PREFIX_RE = re.compile(r"\b(?:NSE|BSE|NASDAQ|NYSE|LSE|TSX):([A-Za-z0-9\.\_]{1,12})\b", re.IGNORECASE)

    # Regex for Yahoo-style suffix tickers (RELIANCE.NS, TCS.BO)
    SUFFIX_TICKER_RE = re.compile(r"\b([A-Za-z0-9]{2,12})\.(?:NS|BO|O|N)\b", re.IGNORECASE)

    # Regex for standalone uppercase tickers (2-6 letters)
    STANDALONE_UPPER_RE = re.compile(r"\b([A-Z]{2,6})\b")

    @classmethod
    def extract_tickers(cls, text: str) -> List[str]:
        """Extracts unique stock tickers from text with deduplication and normalization.

        Returns:
            List of normalized ticker symbols (e.g. ['TCS', 'RELIANCE', 'AAPL']).
        """
        if not text:
            return []

        detected: Set[str] = set()

        # 1. Cashtags (highest precision, e.g. $TCS, $AAPL)
        for match in cls.CASHTAG_RE.finditer(text):
            sym = match.group(1).upper()
            if sym not in TICKER_STOPWORDS:
                detected.add(sym)

        # 2. Exchange prefix (e.g. NSE:RELIANCE)
        for match in cls.EXCHANGE_PREFIX_RE.finditer(text):
            sym = match.group(1).upper()
            if sym not in TICKER_STOPWORDS:
                detected.add(sym)

        # 3. Suffix tickers (e.g. TCS.NS)
        for match in cls.SUFFIX_TICKER_RE.finditer(text):
            sym = match.group(1).upper()
            if sym not in TICKER_STOPWORDS:
                detected.add(sym)

        # 4. Dictionary matching on company names
        lower_text = text.lower()
        for company_name, ticker in COMMON_COMPANY_TICKERS.items():
            # Use boundary match for company name
            pattern = rf"\b{re.escape(company_name)}\b"
            if re.search(pattern, lower_text):
                detected.add(ticker)

        # 5. Standalone uppercase ticker symbols if financial context is present
        # Only check standalone uppercase if we found 0 cashtags to avoid noise
        if not detected:
            for match in cls.STANDALONE_UPPER_RE.finditer(text):
                sym = match.group(1)
                # Check if this uppercase word is a known ticker
                if sym in COMMON_COMPANY_TICKERS.values() and sym not in TICKER_STOPWORDS:
                    detected.add(sym)

        return sorted(detected)

    @classmethod
    def associate_sentences_to_entities(
        cls,
        sentences: List[str],
        default_ticker: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Maps each detected ticker to the list of sentences referencing it.

        If a sentence has no explicit ticker and a default_ticker is provided,
        it is attributed to the default_ticker.
        """
        mapping: Dict[str, List[str]] = {}

        for sentence in sentences:
            tickers = cls.extract_tickers(sentence)
            if not tickers and default_ticker:
                tickers = [default_ticker]

            for ticker in tickers:
                if ticker not in mapping:
                    mapping[ticker] = []
                mapping[ticker].append(sentence)

        return mapping
