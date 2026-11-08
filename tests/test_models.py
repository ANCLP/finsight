"""Tests for baseline model training, probability outputs, and serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.types import DatasetItem, SentimentLabel


def test_baseline_fit_and_predict(sample_dataset_items: list[DatasetItem]):
    texts = [i.text for i in sample_dataset_items]
    labels = [i.sentiment.value for i in sample_dataset_items]

    model = BaselineSentimentModel()
    model.fit(texts, labels)

    assert model.is_ready()

    # Predict positive
    bull_score = model.predict("Operating profit rose significantly beating all forecasts.")
    assert bull_score.label == SentimentLabel.BULLISH
    assert bull_score.bull_bear_score > 0.0
    assert "bullish" in bull_score.probabilities

    # Predict negative
    bear_score = model.predict("Operating loss widened and the company contracted its workforce.")
    assert bear_score.label == SentimentLabel.BEARISH
    assert bear_score.bull_bear_score < 0.0

    # Batch prediction
    batch_scores = model.predict_batch([
        "Strong revenue growth reported.",
        "Severe margin drop and layoffs announced.",
    ])
    assert len(batch_scores) == 2
    assert batch_scores[0].label == SentimentLabel.BULLISH
    assert batch_scores[1].label == SentimentLabel.BEARISH


def test_baseline_save_and_load(sample_dataset_items: list[DatasetItem]):
    texts = [i.text for i in sample_dataset_items]
    labels = [i.sentiment.value for i in sample_dataset_items]

    model = BaselineSentimentModel()
    model.fit(texts, labels)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "test_baseline.joblib"
        model.save(model_file)
        assert model_file.exists()

        # Load fresh model
        loaded_model = BaselineSentimentModel(model_path=model_file)
        assert loaded_model.is_ready()

        score = loaded_model.predict("Operating profit rose by 25%.")
        assert score.label == SentimentLabel.BULLISH
