"""Tests for domain data types and Bull/Bear scoring calculations."""

from __future__ import annotations

import pytest
from fin_sentiment.types import (
    ConfidenceLevel,
    ImpactGrade,
    RawLabel,
    SentimentLabel,
    SentimentScore,
)


def test_raw_label_mapping():
    assert RawLabel.POSITIVE.to_sentiment_label() == SentimentLabel.BULLISH
    assert RawLabel.NEGATIVE.to_sentiment_label() == SentimentLabel.BEARISH
    assert RawLabel.NEUTRAL.to_sentiment_label() == SentimentLabel.NEUTRAL


def test_sentiment_score_from_probabilities_bullish():
    probs = {"bullish": 0.85, "neutral": 0.10, "bearish": 0.05}
    score = SentimentScore.from_probabilities(probs)

    assert score.label == SentimentLabel.BULLISH
    assert score.confidence == 0.85
    assert score.bull_bear_score == pytest.approx(0.80, abs=0.01)
    assert score.impact_grade == ImpactGrade.HIGH
    assert score.confidence_level == ConfidenceLevel.VERY_HIGH


def test_sentiment_score_from_probabilities_bearish():
    probs = {"bullish": 0.05, "neutral": 0.15, "bearish": 0.80}
    score = SentimentScore.from_probabilities(probs)

    assert score.label == SentimentLabel.BEARISH
    assert score.confidence == 0.80
    assert score.bull_bear_score == pytest.approx(-0.75, abs=0.01)
    assert score.impact_grade == ImpactGrade.HIGH
    assert score.confidence_level == ConfidenceLevel.HIGH


def test_sentiment_score_with_catalyst_boost():
    probs = {"bullish": 0.60, "neutral": 0.30, "bearish": 0.10}
    # Initial score: 0.60 - 0.10 = 0.50
    # Boost: +0.15 -> 0.65
    score = SentimentScore.from_probabilities(probs, catalyst_boost=0.15)
    assert score.bull_bear_score == pytest.approx(0.65, abs=0.01)
