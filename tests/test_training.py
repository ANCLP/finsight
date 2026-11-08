"""Tests for training pipeline and metrics calculations."""

from __future__ import annotations

import tempfile
from pathlib import Path
from fin_sentiment.training.baseline_trainer import BaselineTrainer
from fin_sentiment.training.metrics import FinancialMetrics
from fin_sentiment.types import DatasetItem, RawLabel


def test_financial_metrics_calculation():
    y_true = ["positive", "positive", "negative", "neutral", "negative"]
    y_pred = ["positive", "neutral", "negative", "neutral", "positive"]

    metrics = FinancialMetrics.evaluate(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "directional_accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics


def test_baseline_trainer(sample_dataset_items: list[DatasetItem]):
    # Duplicate sample items so there's enough data to split across train/val/test
    extended_items = sample_dataset_items * 6  # 54 items

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "baseline_out.joblib"
        trainer = BaselineTrainer()
        report = trainer.train(
            extended_items,
            test_size=0.20,
            val_size=0.20,
            save_path=save_file,
            random_state=42,
        )

        assert report["model_type"] == "Baseline_TFIDF_LogisticRegression"
        assert save_file.exists()
        assert "accuracy" in report["test_metrics"]
        assert "macro_f1" in report["test_metrics"]
        assert report["samples"]["total"] == len(extended_items)
