"""Unit tests for multi-source dataset integration, synthetic generation, and rebalancing."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import pandas as pd
import pytest

from fin_sentiment.ingestion.dataset_integrator import DatasetIntegrator
from fin_sentiment.ingestion.synthetic_stock_generator import SyntheticStockGenerator
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.training.baseline_trainer import BaselineTrainer
from fin_sentiment.types import DatasetItem, RawLabel


def test_normalize_label_variants():
    # Numeric continuous scores (e.g. FiQA)
    assert DatasetIntegrator.normalize_label(0.75) == RawLabel.POSITIVE
    assert DatasetIntegrator.normalize_label(-0.45) == RawLabel.NEGATIVE
    assert DatasetIntegrator.normalize_label(0.02) == RawLabel.NEUTRAL

    # String aliases
    assert DatasetIntegrator.normalize_label("bullish") == RawLabel.POSITIVE
    assert DatasetIntegrator.normalize_label("BULL") == RawLabel.POSITIVE
    assert DatasetIntegrator.normalize_label("bearish") == RawLabel.NEGATIVE
    assert DatasetIntegrator.normalize_label("SELL") == RawLabel.NEGATIVE
    assert DatasetIntegrator.normalize_label("neutral") == RawLabel.NEUTRAL
    assert DatasetIntegrator.normalize_label("invalid_unknown") is None


def test_synthetic_stock_generator():
    items = SyntheticStockGenerator.generate_dataset(n_samples=300, random_state=42)
    assert len(items) == 300

    pos_count = sum(1 for i in items if i.sentiment == RawLabel.POSITIVE)
    neg_count = sum(1 for i in items if i.sentiment == RawLabel.NEGATIVE)
    neu_count = sum(1 for i in items if i.sentiment == RawLabel.NEUTRAL)

    # Check reasonable ratio distribution
    assert pos_count > 70
    assert neg_count > 100  # Elevated negative ratio to cure deficit
    assert neu_count > 40

    # Ensure prominent tickers are present in generated text
    all_text = " ".join(i.text for i in items)
    assert any(t in all_text for t in ["$AAPL", "$NVDA", "$MSFT", "$RELIANCE", "$TCS", "$INFY"])


def test_dataframe_and_csv_ingestion():
    data = {
        "headline": [
            "Tesla deliveries topped record high in third quarter",
            "Regulatory probe initiated into company accounting practices",
            "Company to hold investor conference call next week",
        ],
        "sentiment": ["bullish", "bearish", "neutral"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_stock_news.csv"
        df.to_csv(csv_path, index=False)

        items = DatasetIntegrator.load_csv(csv_path)
        assert len(items) == 3
        assert items[0].sentiment == RawLabel.POSITIVE
        assert items[1].sentiment == RawLabel.NEGATIVE
        assert items[2].sentiment == RawLabel.NEUTRAL


def test_load_json():
    json_content = """[
        {"text": "Apple reported record services revenue of $22B", "label": "positive"},
        {"text": "Boeing faces new safety review over factory defects", "label": "negative"}
    ]"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "test_news.json"
        json_path.write_text(json_content, encoding="utf-8")

        items = DatasetIntegrator.load_json(json_path)
        assert len(items) == 2
        assert items[0].sentiment == RawLabel.POSITIVE
        assert items[1].sentiment == RawLabel.NEGATIVE


def test_deduplicate_dataset():
    items = [
        DatasetItem(text="Nvidia surged on AI demand", sentiment=RawLabel.POSITIVE),
        DatasetItem(text="Nvidia surged on AI demand", sentiment=RawLabel.POSITIVE),
        DatasetItem(text="  NVIDIA  surged  on  AI  demand  ", sentiment=RawLabel.POSITIVE),
        DatasetItem(text="Apple revenue declined 5%", sentiment=RawLabel.NEGATIVE),
    ]

    deduped = DatasetIntegrator.deduplicate(items)
    assert len(deduped) == 2
    assert deduped[0].text == "Nvidia surged on AI demand"
    assert deduped[1].text == "Apple revenue declined 5%"


def test_rebalance_classes_balanced():
    # Heavily imbalanced sample
    items = (
        [DatasetItem(text=f"Bull {i}", sentiment=RawLabel.POSITIVE) for i in range(10)]
        + [DatasetItem(text=f"Bear {i}", sentiment=RawLabel.NEGATIVE) for i in range(2)]
        + [DatasetItem(text=f"Neut {i}", sentiment=RawLabel.NEUTRAL) for i in range(30)]
    )

    rebalanced = DatasetIntegrator.rebalance_classes(items, strategy="balanced", target_per_class=15)
    assert len(rebalanced) == 45
    assert sum(1 for i in rebalanced if i.sentiment == RawLabel.POSITIVE) == 15
    assert sum(1 for i in rebalanced if i.sentiment == RawLabel.NEGATIVE) == 15
    assert sum(1 for i in rebalanced if i.sentiment == RawLabel.NEUTRAL) == 15


def test_decompose_articles():
    article = {
        "title": "Nvidia Earnings Surge While Intel Lags",
        "body": "Nvidia reported net income rose 600% on massive generative AI data center demand. Operating margin expanded significantly. In contrast, Intel lost market share in the personal computer segment.",
        "sentiment": "bullish",
    }

    decomposed = DatasetIntegrator.decompose_articles([article])
    assert len(decomposed) >= 3
    # First item should be the headline
    assert decomposed[0].text == "Nvidia Earnings Surge While Intel Lags"
    assert all(i.sentiment == RawLabel.POSITIVE for i in decomposed)


def test_expanded_training_and_latency_guarantee():
    # Generate balanced synthetic items
    synthetic_items = SyntheticStockGenerator.generate_dataset(n_samples=180, random_state=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "test_expanded_model.joblib"
        trainer = BaselineTrainer(max_features=5000)
        report = trainer.train(
            synthetic_items,
            test_size=0.20,
            val_size=0.20,
            save_path=save_file,
            rebalance_train_classes=True,
            random_state=42,
        )

        assert save_file.exists()
        assert report["test_metrics"]["accuracy"] > 0.70

        # Verify inference latency guarantee (< 1.5ms per sentence on CPU)
        model = BaselineSentimentModel(model_path=save_file)
        test_sentences = ["Nvidia surged on AI chip sales.", "Company missed earnings and cut guidance."] * 50

        t0 = time.perf_counter()
        _ = model.predict_batch(test_sentences)
        elapsed = time.perf_counter() - t0
        latency_per_sample_ms = (elapsed / len(test_sentences)) * 1000

        assert latency_per_sample_ms < 1.5, f"Latency was {latency_per_sample_ms:.2f}ms (must be < 1.5ms)"
