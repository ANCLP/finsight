"""Abstract base class and protocol for financial sentiment models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from fin_sentiment.types import SentimentScore


class BaseSentimentModel(ABC):
    """Abstract interface that all sentiment classification backends must implement."""

    @abstractmethod
    def predict(self, text: str) -> SentimentScore:
        """Classifies sentiment for a single financial text snippet.

        Returns:
            SentimentScore containing label, confidence, bull_bear_score, and probability distribution.
        """
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Classifies sentiment for a batch of financial texts efficiently.

        Returns:
            List of SentimentScore objects.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Checks if the model is fitted/loaded and ready for inference."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Serializes the model weights/pipeline to disk."""
        raise NotImplementedError("Serialization not implemented for this model backend.")

    def load(self, path: Union[str, Path]) -> None:
        """Deserializes model weights/pipeline from disk."""
        raise NotImplementedError("Loading not implemented for this model backend.")
