"""Comprehensive financial NLP evaluation metrics.

Calculates:
- Standard Classification Metrics: Accuracy, Balanced Accuracy, Macro F1, Weighted F1
- Directional Accuracy: Binary precision between Bullish vs Bearish (filtering out neutral)
- Confusion Matrix & Per-Class Breakdown
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fin_sentiment.types import RawLabel, SentimentLabel


class FinancialMetrics:
    """Computes specialized metrics for financial text sentiment models."""

    @classmethod
    def evaluate(
        cls,
        y_true: List[Union[str, RawLabel, SentimentLabel]],
        y_pred: List[Union[str, RawLabel, SentimentLabel]],
    ) -> Dict[str, Any]:
        """Evaluates model predictions against ground truth labels.

        Args:
            y_true: Ground truth labels (e.g. ['positive', 'negative', 'neutral'] or ['bullish', ...])
            y_pred: Predicted labels

        Returns:
            Dictionary containing accuracy, macro_f1, weighted_f1, directional_accuracy,
            per_class metrics, and confusion matrix.
        """
        # Normalize labels to canonical lowercase strings
        true_norm = [cls._normalize_label(y) for y in y_true]
        pred_norm = [cls._normalize_label(y) for y in y_pred]

        classes = ["negative", "neutral", "positive"]

        acc = float(accuracy_score(true_norm, pred_norm))
        bal_acc = float(balanced_accuracy_score(true_norm, pred_norm))
        macro_f1 = float(f1_score(true_norm, pred_norm, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(true_norm, pred_norm, average="weighted", zero_division=0))

        # Per-class metrics
        precisions = precision_score(true_norm, pred_norm, labels=classes, average=None, zero_division=0)
        recalls = recall_score(true_norm, pred_norm, labels=classes, average=None, zero_division=0)
        f1s = f1_score(true_norm, pred_norm, labels=classes, average=None, zero_division=0)

        per_class = {}
        for idx, c in enumerate(classes):
            trading_alias = "bearish" if c == "negative" else ("bullish" if c == "positive" else "neutral")
            per_class[trading_alias] = {
                "precision": round(float(precisions[idx]), 4),
                "recall": round(float(recalls[idx]), 4),
                "f1": round(float(f1s[idx]), 4),
            }

        # Directional accuracy: When truth is polar (positive or negative), did model get direction right?
        polar_indices = [i for i, y in enumerate(true_norm) if y in ("positive", "negative")]
        if polar_indices:
            polar_true = [true_norm[i] for i in polar_indices]
            polar_pred = [pred_norm[i] for i in polar_indices]
            # Match is positive-positive or negative-negative
            dir_correct = sum(1 for t, p in zip(polar_true, polar_pred) if t == p)
            directional_accuracy = round(dir_correct / len(polar_indices), 4)
        else:
            directional_accuracy = 0.0

        # Confusion Matrix
        cm = confusion_matrix(true_norm, pred_norm, labels=classes)

        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "directional_accuracy": directional_accuracy,
            "per_class": per_class,
            "confusion_matrix": {
                "classes": ["bearish", "neutral", "bullish"],
                "matrix": cm.tolist(),
            },
            "total_samples": len(y_true),
        }

    @staticmethod
    def _normalize_label(label: Union[str, RawLabel, SentimentLabel]) -> str:
        """Normalizes any label representation to standard 'positive', 'neutral', 'negative'."""
        val = label.value if hasattr(label, "value") else str(label)
        val = val.lower().strip()
        if val in ("positive", "bullish", "bull"):
            return "positive"
        if val in ("negative", "bearish", "bear"):
            return "negative"
        return "neutral"
