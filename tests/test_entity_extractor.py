"""Tests for stock ticker extraction and financial catalyst detection."""

from __future__ import annotations

from fin_sentiment.entity.financial_catalysts import CatalystExtractor
from fin_sentiment.entity.ticker_extractor import TickerExtractor
from fin_sentiment.types import CatalystType, SentimentLabel


def test_extract_tickers_cashtags():
    text = "Traders are buying $AAPL and $NVDA ahead of earnings, while dumping $TSLA."
    tickers = TickerExtractor.extract_tickers(text)
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "TSLA" in tickers


def test_extract_tickers_exchange_prefixed():
    text = "Massive block deal observed in NSE:RELIANCE and BSE:500325."
    tickers = TickerExtractor.extract_tickers(text)
    assert "RELIANCE" in tickers
    assert "500325" in tickers


def test_extract_tickers_company_names():
    text = "Tata Consultancy Services won a mega contract, while Infosys shares fell."
    tickers = TickerExtractor.extract_tickers(text)
    assert "TCS" in tickers
    assert "INFY" in tickers


def test_detect_bullish_catalysts():
    text = "Reliance net profit surged 28% beating estimates as refining margins expanded."
    catalysts = CatalystExtractor.detect_catalysts(text)
    types = [c.catalyst for c in catalysts]
    assert CatalystType.EARNINGS_BEAT in types
    assert CatalystType.MARGIN_EXPANSION in types

    boost = CatalystExtractor.compute_catalyst_boost(catalysts)
    assert boost > 0.0


def test_detect_bearish_catalysts():
    text = "The company missed revenue forecasts and announced plans to lay off 400 workers."
    catalysts = CatalystExtractor.detect_catalysts(text)
    types = [c.catalyst for c in catalysts]
    assert CatalystType.EARNINGS_MISS in types
    assert CatalystType.LAYOFF_RESTRUCTURING in types

    boost = CatalystExtractor.compute_catalyst_boost(catalysts)
    assert boost < 0.0


def test_extract_tickers_mixed_cashtag_and_standalone():
    text = "$TCS reported record gains while INFY and MSFT dropped."
    tickers = TickerExtractor.extract_tickers(text)
    assert "TCS" in tickers
    assert "INFY" in tickers
    assert "MSFT" in tickers


def test_extract_tickers_ignores_monetary_dollar_amounts():
    text = "Apple ($AAPL) posted revenue of $89.5 billion and declared a dividend of $0.24 per share."
    tickers = TickerExtractor.extract_tickers(text)
    assert tickers == ["AAPL"]
    assert "89.5" not in tickers
    assert "0.24" not in tickers
