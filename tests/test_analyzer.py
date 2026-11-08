"""Tests for the high-level FinancialNewsSentimentAnalyzer facade."""

from __future__ import annotations

from fin_sentiment.analyzer import FinancialNewsSentimentAnalyzer
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.types import DatasetItem, SentimentLabel


def test_analyzer_single_text(sample_dataset_items: list[DatasetItem]):
    texts = [i.text for i in sample_dataset_items]
    labels = [i.sentiment.value for i in sample_dataset_items]

    baseline = BaselineSentimentModel().fit(texts, labels)
    analyzer = FinancialNewsSentimentAnalyzer(model=baseline)

    score = analyzer.analyze_text("Operating profit rose significantly beating consensus.")
    assert score.label == SentimentLabel.BULLISH
    assert score.bull_bear_score > 0.0
    assert 0.0 <= score.confidence <= 1.0


def test_analyzer_article_with_entities(sample_dataset_items: list[DatasetItem]):
    texts = [i.text for i in sample_dataset_items]
    labels = [i.sentiment.value for i in sample_dataset_items]

    baseline = BaselineSentimentModel().fit(texts, labels)
    analyzer = FinancialNewsSentimentAnalyzer(model=baseline)

    article_text = (
        "Tech Giants Diverge in Quarterly Earnings\n\n"
        "Tata Consultancy Services ($TCS) reported operating profit rose 25% beating all estimates. "
        "However, Infosys ($INFY) suffered a steep margin contraction and announced restructuring plans. "
        "Both stocks saw heavy trading volumes."
    )

    analysis = analyzer.analyze_article(article_text)
    assert len(analysis.sentences) >= 3
    assert "TCS" in analysis.entities
    assert "INFY" in analysis.entities

    # TCS sentence was positive, so TCS sentiment should be bullish
    assert analysis.entities["TCS"].sentiment.label == SentimentLabel.BULLISH
    assert analysis.entities["TCS"].sentiment.bull_bear_score > 0.0

    # INFY sentence was negative, so INFY sentiment should be bearish
    assert analysis.entities["INFY"].sentiment.label == SentimentLabel.BEARISH
    assert analysis.entities["INFY"].sentiment.bull_bear_score < 0.0

    # Check catalysts were picked up
    assert len(analysis.catalysts) > 0
    assert len(analysis.summary) > 0
