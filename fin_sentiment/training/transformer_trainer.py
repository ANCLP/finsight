"""Fine-tuning pipeline template for Transformer architectures (FinBERT / RoBERTa).

Provides:
- PyTorch / HuggingFace Dataset encapsulation
- Stratified splitting & label encoding
- Configurable fine-tuning hyperparameters
- Checkpoint persistence and metrics logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.training.metrics import FinancialMetrics
from fin_sentiment.types import DatasetItem, RawLabel

logger = logging.getLogger(__name__)

LABEL_TO_ID = {
    RawLabel.POSITIVE.value: 0,
    RawLabel.NEGATIVE.value: 1,
    RawLabel.NEUTRAL.value: 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class TransformerTrainer:
    """Template and executor for fine-tuning transformer models on financial sentiment datasets."""

    def __init__(
        self,
        base_model_name: str = "ProsusAI/finbert",
        output_dir: str = "checkpoints/finbert_finetuned",
        learning_rate: float = 2e-5,
        epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 128,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = True,
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.fp16 = fp16

    def train(
        self,
        items: List[DatasetItem],
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Runs the transformer fine-tuning workflow on the given dataset items."""
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:
            raise ImportError(
                "PyTorch and HuggingFace Transformers are required for TransformerTrainer. "
                "Install them via: pip install 'fin-sentiment[torch]'"
            ) from e

        loader = PhraseBankLoader()
        train_items, val_items, test_items = loader.create_splits(
            items,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )

        # PyTorch Dataset implementation
        class PhraseDataset(torch.utils.data.Dataset):
            def __init__(self, data_items: List[DatasetItem]):
                self.texts = [i.text for i in data_items]
                self.labels = [LABEL_TO_ID[i.sentiment.value] for i in data_items]
                self.encodings = tokenizer(
                    self.texts,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                )

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        train_dataset = PhraseDataset(train_items)
        val_dataset = PhraseDataset(val_items)
        test_dataset = PhraseDataset(test_items)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            pred_strs = [ID_TO_LABEL[p] for p in preds]
            true_strs = [ID_TO_LABEL[l] for l in labels]
            res = FinancialMetrics.evaluate(true_strs, pred_strs)
            return {
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "directional_accuracy": res["directional_accuracy"],
            }

        cuda_available = torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=cuda_available and self.fp16,
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting transformer fine-tuning on %d samples...", len(train_dataset))
        trainer.train()

        # Final evaluation on holdout test set
        test_results = trainer.evaluate(test_dataset)
        logger.info("Test set evaluation results: %s", test_results)

        # Save model and tokenizer
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        logger.info("Fine-tuned transformer saved to %s", self.output_dir)

        return {
            "model_type": f"Transformer_{self.base_model_name}",
            "checkpoint_dir": str(self.output_dir.resolve()),
            "samples": {
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            },
            "test_metrics": test_results,
        }
