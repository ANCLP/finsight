"""Comprehensive tests validating remediation of the 9 architectural red flags."""

from unittest.mock import patch, MagicMock
import pytest

from fin_sentiment.analyzer import FinancialNewsSentimentAnalyzer
from fin_sentiment.models.llm_arbiter import GeminiFinancialArbiter
from fin_sentiment.types import CatalystType, SentimentLabel


@pytest.fixture
def analyzer():
    return FinancialNewsSentimentAnalyzer()


def test_contrastive_clause_remediation(analyzer):
    """Red Flag 1: 'Although revenue surged 32% beating targets, unexpected liabilities wiped out cash flow.'"""
    headline = "Although revenue surged 32% beating Wall Street targets, the firm reported severe unexpected litigation liabilities that completely wiped out operating cash flow."
    score = analyzer.analyze_text(headline)
    # Operative clause (liabilities wiped out cash flow) must dominate over concession (revenue surged)
    assert score.label == SentimentLabel.BEARISH
    assert score.bull_bear_score < 0


def test_directional_metric_debt_expansion(analyzer):
    """Red Flag 5: 'expanded debt' must evaluate as Bearish."""
    headline = "Company significantly expanded debt borrowings and liabilities this quarter."
    score = analyzer.analyze_text(headline)
    assert score.label == SentimentLabel.BEARISH
    assert score.bull_bear_score < 0


def test_directional_metric_deleveraging(analyzer):
    """Red Flag 5: 'reduced debt' must evaluate as Bullish."""
    headline = "Company successfully repaid debt obligations and reduced borrowings."
    score = analyzer.analyze_text(headline)
    assert score.label == SentimentLabel.BULLISH
    assert score.bull_bear_score > 0


def test_scare_quotes_attenuation(analyzer):
    """Red Flag 7: Scare quotes around superlatives ('stellar', 'revolutionary') must be attenuated."""
    headline = "CEO touts 'stellar' and 'revolutionary' turnaround amid delivery miss."
    score = analyzer.analyze_text(headline)
    # The score should be subdued/neutral rather than high-conviction bullish
    assert score.probabilities.get("bullish", 0.0) < 0.60


def test_rumor_denial_neutralization(analyzer):
    """Red Flag 4: 'Management denied reports of probe' should not trigger panic bearish catalyst."""
    headline = "Management vehemently denied media reports of an accounting fraud probe."
    score = analyzer.analyze_text(headline)
    # Denial should neutralize the severe catalyst boost
    assert score.bull_bear_score > -0.40


def test_zero_sentiment_auditor_resignation(analyzer):
    """Red Flag 3: 'Auditor resigns' is an implicit distress catalyst."""
    headline = "Statutory auditor resigns citing lack of financial information."
    score = analyzer.analyze_text(headline)
    assert score.label == SentimentLabel.BEARISH
    assert score.bull_bear_score <= -0.15


def test_gemini_arbiter_should_arbitrate():
    """Validates arbitration trigger rules for Gemini 1.5 Flash."""
    arbiter = GeminiFinancialArbiter(api_key="fake_test_key_1234567890")

    # Multi-entity headline
    assert arbiter.should_arbitrate("Tata Motors gains as Maruti loses market share", extracted_entities=["TATAMOTORS", "MARUTI"]) is True

    # Consensus comparison
    assert arbiter.should_arbitrate("Revenues rose 15% but missed consensus expectations") is True

    # Regulatory distress
    assert arbiter.should_arbitrate("Auditor resigns pending forensic audit") is True

    # Simple unambiguous headline
    assert arbiter.should_arbitrate("TCS signs $500M contract with European bank", extracted_entities=["TCS"], baseline_conf=0.88) is False


def test_gemini_arbiter_integration_mock(analyzer):
    """Tests that analyzer uses Gemini Arbiter response when available."""
    mock_llm_response = {
        "label": "bearish",
        "bull_bear_score": -0.72,
        "confidence": 0.90,
        "reasoning": "Missed Wall Street consensus estimates despite top-line growth.",
        "primary_catalyst": "EARNINGS_MISS",
        "entity_sentiments": {"TSLA": -0.72},
        "source": "gemini_1.5_flash",
    }

    with patch.object(analyzer.arbiter, "is_available", return_value=True):
        with patch.object(analyzer.arbiter, "should_arbitrate", return_value=True):
            with patch.object(analyzer.arbiter, "analyze", return_value=mock_llm_response):
                score = analyzer.analyze_text("Tesla revenues up 20% YoY, but missed consensus views of $25B.")
                assert score.label == SentimentLabel.BEARISH
                assert score.bull_bear_score == -0.72
                assert score.confidence == 0.90


def test_company_expands_debt_obligations(analyzer):
    """Protects against regression for 'Company expands debt obligations'."""
    score = analyzer.analyze_text("Company expands debt obligations")
    assert score.label == SentimentLabel.BEARISH
    assert score.bull_bear_score <= -0.40
    assert score.probabilities[SentimentLabel.BEARISH] > 0.70

