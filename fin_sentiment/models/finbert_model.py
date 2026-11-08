"""Transformer-based financial sentiment classifier (FinBERT adapter).

Supports:
- Pre-trained models: ProsusAI/finbert, yiyanghkust/finbert-tone
- Local fine-tuned checkpoint loading
- Automatic hardware acceleration (NVIDIA CUDA / CPU)
- Half-precision FP16 evaluation on CUDA
- High-throughput batch inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from fin_sentiment.models.base import BaseSentimentModel
from fin_sentiment.types import SentimentScore

logger = logging.getLogger(__name__)

DEFAULT_FINBERT_REPO = "ProsusAI/finbert"


class FinBertSentimentModel(BaseSentimentModel):
    """Deep learning financial sentiment analyzer based on FinBERT architecture."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_FINBERT_REPO,
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 128,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self._tokenizer = None
        self._model = None
        self._device = device
        self._ready = False

    def is_ready(self) -> bool:
        """Returns True if the PyTorch model and tokenizer are initialized."""
        return self._ready and self._model is not None and self._tokenizer is not None

    def initialize(self) -> "FinBertSentimentModel":
        """Loads the transformer model and tokenizer into memory."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "PyTorch and HuggingFace Transformers are required for FinBertSentimentModel. "
                "Install them via: pip install 'fin-sentiment[torch]'"
            ) from e

        # Resolve device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Initializing FinBERT model from '%s' on %s...", self.model_name_or_path, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)

        if self._device == "cuda" and self.use_fp16:
            self._model = self._model.half()

        self._model.to(self._device)
        self._model.eval()
        self._ready = True

        logger.info("FinBERT initialized successfully.")
        return self

    def predict(self, text: str) -> SentimentScore:
        """Classifies sentiment for a single financial text."""
        scores = self.predict_batch([text])
        return scores[0]

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[SentimentScore]:
        """Classifies sentiment for a list of texts in batches."""
        if not self.is_ready():
            self.initialize()

        import torch

        if not texts:
            return []

        results: List[SentimentScore] = []
        id2label = getattr(self._model.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for p_row in probs:
                prob_map: Dict[str, float] = {}
                for idx, p_val in enumerate(p_row):
                    raw_cls = id2label.get(idx, str(idx)).lower()
                    if "pos" in raw_cls or "bull" in raw_cls:
                        prob_map["bullish"] = float(p_val)
                    elif "neg" in raw_cls or "bear" in raw_cls:
                        prob_map["bearish"] = float(p_val)
                    else:
                        prob_map["neutral"] = float(p_val)

                score = SentimentScore.from_probabilities(prob_map)
                results.append(score)

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Saves the fine-tuned model and tokenizer to a directory."""
        if not self.is_ready():
            raise RuntimeError("Cannot save an uninitialized model.")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out_dir)
        self._tokenizer.save_pretrained(out_dir)
        logger.info("Saved FinBERT model to %s", out_dir)

    def load(self, path: Union[str, Path]) -> None:
        """Loads fine-tuned weights from a local directory."""
        self.model_name_or_path = str(path)
        self.initialize()
