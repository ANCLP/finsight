"""Training pipeline for baseline financial sentiment models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.training.metrics import FinancialMetrics
from fin_sentiment.types import DatasetItem

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """Orchestrates end-to-end training and evaluation of baseline TF-IDF + Classifier models."""

    def __init__(
        self,
        c_param: float = 2.5,
        ngram_range: tuple = (1, 3),
    ):
        self.c_param = c_param
        self.ngram_range = ngram_range
        self.model: Optional[BaselineSentimentModel] = None

    def train(
        self,
        items: List[DatasetItem],
        test_size: float = 0.15,
        val_size: float = 0.15,
        save_path: Optional[Union[str, Path]] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Runs the complete training and evaluation pipeline on provided DatasetItems.

        Returns:
            Dictionary with training summary, validation metrics, and test metrics.
        """
        loader = PhraseBankLoader()
        train_items, val_items, test_items = loader.create_splits(
            items,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        train_texts = [i.text for i in train_items]
        train_labels = [i.sentiment.value for i in train_items]

        # Initialize and fit model
        self.model = BaselineSentimentModel(
            c_param=self.c_param,
            ngram_range=self.ngram_range,
        )
        self.model.fit(train_texts, train_labels)

        # Evaluate on Validation Set
        val_texts = [i.text for i in val_items]
        val_true = [i.sentiment.value for i in val_items]
        val_scores = self.model.predict_batch(val_texts)
        val_pred = [s.label.value for s in val_scores]
        val_metrics = FinancialMetrics.evaluate(val_true, val_pred)

        # Evaluate on Test Set
        test_texts = [i.text for i in test_items]
        test_true = [i.sentiment.value for i in test_items]
        test_scores = self.model.predict_batch(test_texts)
        test_pred = [s.label.value for s in test_scores]
        test_metrics = FinancialMetrics.evaluate(test_true, test_pred)

        logger.info(
            "Test Results -> Accuracy: %.2f%% | Macro F1: %.2f | Directional Accuracy: %.2f%%",
            test_metrics["accuracy"] * 100,
            test_metrics["macro_f1"],
            test_metrics["directional_accuracy"] * 100,
        )

        # Save model if requested
        saved_file = None
        if save_path:
            out = Path(save_path)
            self.model.save(out)
            saved_file = str(out.resolve())

        return {
            "model_type": "Baseline_TFIDF_LogisticRegression",
            "samples": {
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
                "total": len(items),
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "saved_model_path": saved_file,
        }
