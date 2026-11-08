"""Ingestion and dataset loading utilities for Financial PhraseBank and financial news CSVs.

Supports:
- all-data.csv (Standard CSV format: sentiment,sentence)
- Sentences_*Agree.txt (Official Aalto University format: sentence@sentiment)
- Robust multi-encoding fallback (latin-1, utf-8, iso-8859-1)
- Stratified train/val/test splits preserving sentiment class balance
- Agreement consensus filtering (50%, 66%, 75%, 100%)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from fin_sentiment.types import DatasetItem, RawLabel

logger = logging.getLogger(__name__)

SUPPORTED_ENCODINGS = ["latin-1", "iso-8859-1", "utf-8", "cp1252"]

AGREEMENT_FILES = {
    1.00: "Sentences_AllAgree.txt",
    0.75: "Sentences_75Agree.txt",
    0.66: "Sentences_66Agree.txt",
    0.50: "Sentences_50Agree.txt",
}


class PhraseBankLoader:
    """Robust loader and parser for Financial PhraseBank datasets and financial news CSVs."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """Initialize loader. If data_dir is not provided, defaults to current directory."""
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            # Look in parent or current dir
            self.data_dir = Path(__file__).resolve().parent.parent.parent

    def load_csv(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> List[DatasetItem]:
        """Loads items from `all-data.csv` (format: sentiment,sentence).

        Args:
            filepath: Path to the CSV file. If None, checks data_dir / 'all-data.csv'.

        Returns:
            List of validated DatasetItem objects.
        """
        path = Path(filepath) if filepath else self.data_dir / "all-data.csv"
        if not path.exists():
            raise FileNotFoundError(f"Financial CSV dataset not found at: {path}")

        items: List[DatasetItem] = []
        raw_lines = self._read_file_safe(path)

        # Parse CSV lines
        reader = csv.reader(raw_lines)
        for row_idx, row in enumerate(reader):
            if not row or len(row) < 2:
                continue

            sentiment_str = row[0].strip().lower()
            text_str = row[1].strip()

            # Handle possible header
            if row_idx == 0 and sentiment_str in ("sentiment", "label") and "sentence" in text_str.lower():
                continue

            try:
                raw_label = RawLabel(sentiment_str)
            except ValueError:
                # Invert if user file is sentence,sentiment
                try:
                    alt_label = RawLabel(text_str.lower())
                    text_str = sentiment_str
                    raw_label = alt_label
                except ValueError:
                    logger.debug("Skipping unrecognized sentiment row %d: %s", row_idx, sentiment_str)
                    continue

            if text_str:
                items.append(
                    DatasetItem(
                        text=text_str,
                        sentiment=raw_label,
                        agreement_level=0.50,
                        source_file=path.name,
                    )
                )

        logger.info("Loaded %d items from %s", len(items), path.name)
        return items

    def load_agreement_txt(
        self,
        min_agreement: float = 0.50,
        filepath: Optional[Union[str, Path]] = None,
    ) -> List[DatasetItem]:
        """Loads sentences from the FinancialPhraseBank consensus text files

        (format: sentence@sentiment).

        Args:
            min_agreement: Desired annotator consensus threshold.
                - 1.00: 100% agreement (Sentences_AllAgree.txt, 2264 samples)
                - 0.75: >=75% agreement (Sentences_75Agree.txt, 3453 samples)
                - 0.66: >=66% agreement (Sentences_66Agree.txt, 4217 samples)
                - 0.50: >=50% majority (Sentences_50Agree.txt, 4846 samples)
            filepath: Direct path to a specific text file. If provided, overrides min_agreement selection.

        Returns:
            List of validated DatasetItem objects.
        """
        if filepath is not None:
            path = Path(filepath)
            target_agreement = min_agreement
        else:
            # Map agreement float to nearest valid file
            matched_key = min(AGREEMENT_FILES.keys(), key=lambda k: abs(k - min_agreement))
            target_filename = AGREEMENT_FILES[matched_key]
            target_agreement = matched_key

            # Check inside FinancialPhraseBank/ directory first, then root
            candidate1 = self.data_dir / "FinancialPhraseBank" / target_filename
            candidate2 = self.data_dir / target_filename
            path = candidate1 if candidate1.exists() else candidate2

        if not path.exists():
            raise FileNotFoundError(f"Consensus text file not found at: {path}")

        items: List[DatasetItem] = []
        raw_lines = self._read_file_safe(path)

        for line_idx, line in enumerate(raw_lines):
            line = line.strip()
            if not line:
                continue

            # Standard format: sentence@sentiment
            if "@" in line:
                parts = line.rsplit("@", 1)
                text_str = parts[0].strip()
                sentiment_str = parts[1].strip().lower()
            else:
                continue

            try:
                raw_label = RawLabel(sentiment_str)
            except ValueError:
                continue

            if text_str:
                items.append(
                    DatasetItem(
                        text=text_str,
                        sentiment=raw_label,
                        agreement_level=target_agreement,
                        source_file=path.name,
                    )
                )

        logger.info(
            "Loaded %d items from %s (consensus: %.0f%%)",
            len(items),
            path.name,
            target_agreement * 100,
        )
        return items

    def load_dataset(
        self,
        min_agreement: Optional[float] = None,
        source: str = "auto",
    ) -> List[DatasetItem]:
        """High-level loader that automatically discovers and parses the best available dataset.

        Args:
            min_agreement: If specified, loads from the corresponding consensus text file.
            source: 'auto', 'csv', or 'txt'.

        Returns:
            List of DatasetItem objects.
        """
        if source == "csv":
            return self.load_csv()

        if min_agreement is not None or source == "txt":
            agreement = min_agreement if min_agreement is not None else 0.50
            return self.load_agreement_txt(min_agreement=agreement)

        # Auto detection: prefer AllAgree or 75Agree if available, else csv, else 50Agree
        csv_path = self.data_dir / "all-data.csv"
        if csv_path.exists():
            return self.load_csv(csv_path)

        return self.load_agreement_txt(min_agreement=0.50)

    def to_dataframe(self, items: List[DatasetItem]) -> pd.DataFrame:
        """Converts a list of DatasetItem to a pandas DataFrame with normalized fields."""
        data = [
            {
                "text": item.text,
                "raw_sentiment": item.sentiment.value,
                "sentiment": item.sentiment.to_sentiment_label().value,
                "agreement_level": item.agreement_level,
                "source_file": item.source_file,
            }
            for item in items
        ]
        return pd.DataFrame(data)

    def create_splits(
        self,
        items: List[DatasetItem],
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[List[DatasetItem], List[DatasetItem], List[DatasetItem]]:
        """Generates stratified train, validation, and test splits preserving sentiment distribution.

        Returns:
            (train_items, val_items, test_items)
        """
        if not items:
            raise ValueError("Cannot split an empty dataset.")

        texts = [i.text for i in items]
        labels = [i.sentiment.value for i in items]

        # First split into train_val and test
        train_val_idx, test_idx = train_test_split(
            range(len(items)),
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )

        train_val_items = [items[i] for i in train_val_idx]
        train_val_labels = [labels[i] for i in train_val_idx]
        test_items = [items[i] for i in test_idx]

        # Then split train_val into train and val
        relative_val_size = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            range(len(train_val_items)),
            test_size=relative_val_size,
            stratify=train_val_labels,
            random_state=random_state,
        )

        train_items = [train_val_items[i] for i in train_idx]
        val_items = [train_val_items[i] for i in val_idx]

        logger.info(
            "Split %d items -> Train: %d, Val: %d, Test: %d",
            len(items),
            len(train_items),
            len(val_items),
            len(test_items),
        )
        return train_items, val_items, test_items

    def _read_file_safe(self, path: Path) -> List[str]:
        """Safely reads a text file trying multiple encodings."""
        for enc in SUPPORTED_ENCODINGS:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.readlines()
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback with replacement
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
