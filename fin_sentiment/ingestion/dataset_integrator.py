"""Multi-source financial sentiment dataset integrator and normalizer.

Enables seamless consolidation, cleaning, deduplication, and class balancing across:
- Financial PhraseBank (Aalto University consensus benchmarks)
- Modern Stock News datasets (FiQA 2018, Twitter Financial News, StockTwits)
- Synthetic modern equity catalyst datasets
- Full article decomposition into sentence-level sentiment items
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from fin_sentiment.ingestion.article_parser import ArticleParser
from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.ingestion.synthetic_stock_generator import SyntheticStockGenerator
from fin_sentiment.ingestion.text_preprocessor import TextPreprocessor
from fin_sentiment.types import DatasetItem, RawLabel, SentimentLabel

logger = logging.getLogger(__name__)

# Common column name heuristics for tabular financial datasets
TEXT_COLUMN_CANDIDATES = ["text", "sentence", "headline", "content", "title", "phrase", "news"]
LABEL_COLUMN_CANDIDATES = ["sentiment", "label", "polarity", "target", "score", "class", "rating"]


class DatasetIntegrator:
    """Consolidates and balances heterogeneous financial sentiment datasets."""

    @classmethod
    def normalize_label(cls, val: Any) -> Optional[RawLabel]:
        """Normalizes heterogeneous sentiment formats into RawLabel."""
        if val is None:
            return None

        # Case 1: Numeric continuous score (e.g. FiQA -1.0 to +1.0)
        if isinstance(val, (int, float)):
            if val > 0.10:
                return RawLabel.POSITIVE
            if val < -0.10:
                return RawLabel.NEGATIVE
            return RawLabel.NEUTRAL

        # Case 2: String representations
        s = str(val).strip().lower()

        # Handle numeric strings
        try:
            num = float(s)
            if num > 0.10:
                return RawLabel.POSITIVE
            if num < -0.10:
                return RawLabel.NEGATIVE
            return RawLabel.NEUTRAL
        except ValueError:
            pass

        # String mappings
        positive_aliases = {"positive", "bullish", "bull", "pos", "1", "+1", "buy", "up"}
        negative_aliases = {"negative", "bearish", "bear", "neg", "-1", "sell", "down"}
        neutral_aliases = {"neutral", "neut", "none", "0", "hold"}

        if s in positive_aliases:
            return RawLabel.POSITIVE
        if s in negative_aliases:
            return RawLabel.NEGATIVE
        if s in neutral_aliases:
            return RawLabel.NEUTRAL

        return None

    @classmethod
    def load_csv(
        cls,
        filepath: Union[str, Path],
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> List[DatasetItem]:
        """Loads and normalizes items from an arbitrary financial news CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df = pd.read_csv(path)
        return cls.from_dataframe(df, text_column=text_column, label_column=label_column, source_file=path.name)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> List[DatasetItem]:
        """Constructs DatasetItems from a pandas DataFrame using column discovery heuristics."""
        cols_lower = {col.lower(): col for col in df.columns}

        # Resolve text column
        resolved_text_col = text_column
        if not resolved_text_col:
            for candidate in TEXT_COLUMN_CANDIDATES:
                if candidate in cols_lower:
                    resolved_text_col = cols_lower[candidate]
                    break
        if not resolved_text_col and len(df.columns) >= 2:
            resolved_text_col = df.columns[1]  # common phrasebank format: label, sentence

        # Resolve label column
        resolved_label_col = label_column
        if not resolved_label_col:
            for candidate in LABEL_COLUMN_CANDIDATES:
                if candidate in cols_lower:
                    resolved_label_col = cols_lower[candidate]
                    break
        if not resolved_label_col and len(df.columns) >= 1:
            resolved_label_col = df.columns[0]

        if not resolved_text_col or not resolved_label_col:
            raise ValueError(f"Could not automatically detect text and label columns from: {list(df.columns)}")

        items: List[DatasetItem] = []
        for _, row in df.iterrows():
            raw_text = str(row[resolved_text_col]).strip()
            raw_label_val = row[resolved_label_col]
            norm_label = cls.normalize_label(raw_label_val)

            if raw_text and norm_label and len(raw_text) > 3:
                clean_text = TextPreprocessor.clean_text(raw_text)
                items.append(DatasetItem(
                    text=clean_text,
                    sentiment=norm_label,
                    agreement_level=1.0,
                    source_file=source_file or "dataframe_import",
                ))

        logger.info("Imported %d valid items from DataFrame.", len(items))
        return items

    @classmethod
    def load_json(
        cls,
        filepath: Union[str, Path],
        text_key: str = "sentence",
        label_key: str = "sentiment",
    ) -> List[DatasetItem]:
        """Loads items from a JSON array or JSON Lines (.jsonl) file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found at: {path}")

        items: List[DatasetItem] = []
        content = path.read_text(encoding="utf-8", errors="replace").strip()

        records = []
        if content.startswith("[") and content.endswith("]"):
            records = json.loads(content)
        else:
            # Parse as JSON Lines
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for rec in records:
            if not isinstance(rec, dict):
                continue
            text = rec.get(text_key) or rec.get("text") or rec.get("headline") or ""
            label_val = rec.get(label_key) or rec.get("label") or rec.get("score")
            norm_label = cls.normalize_label(label_val)

            if text and norm_label:
                clean_text = TextPreprocessor.clean_text(str(text))
                items.append(DatasetItem(
                    text=clean_text,
                    sentiment=norm_label,
                    agreement_level=1.0,
                    source_file=path.name,
                ))

        logger.info("Imported %d items from %s", len(items), path.name)
        return items

    @classmethod
    def decompose_articles(
        cls,
        articles: List[Union[str, Dict[str, Any]]],
        default_label: Optional[RawLabel] = None,
    ) -> List[DatasetItem]:
        """Deconstructs full articles into distinct headline and sentence training samples.

        Prevents feature dilution from monolithic 1,000-word articles by preserving
        granular sentence-level boundaries.
        """
        sentence_items: List[DatasetItem] = []

        for art in articles:
            label = default_label
            if isinstance(art, dict):
                label_candidate = art.get("sentiment") or art.get("label")
                if label_candidate:
                    label = cls.normalize_label(label_candidate)

            hl, body, sents = ArticleParser.parse(art)

            # Add headline as distinct high-weight item if available
            if hl and label:
                sentence_items.append(DatasetItem(
                    text=hl,
                    sentiment=label,
                    agreement_level=1.0,
                    source_file="article_headline",
                ))

            # Add constituent sentences
            if label:
                for s in sents:
                    # Skip trivial or very short transitional phrases
                    if len(s.split()) >= 6:
                        sentence_items.append(DatasetItem(
                            text=s,
                            sentiment=label,
                            agreement_level=1.0,
                            source_file="article_sentence",
                        ))

        return sentence_items

    @classmethod
    def deduplicate(cls, items: List[DatasetItem]) -> List[DatasetItem]:
        """Removes exact and normalized duplicate sentences using SHA-256 fingerprinting."""
        seen_hashes = set()
        unique_items: List[DatasetItem] = []

        for item in items:
            normalized_core = " ".join(item.text.lower().split())
            h = hashlib.sha256(normalized_core.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_items.append(item)

        logger.info("Deduplication reduced %d samples to %d unique items.", len(items), len(unique_items))
        return unique_items

    @classmethod
    def rebalance_classes(
        cls,
        items: List[DatasetItem],
        strategy: str = "balanced",
        target_per_class: Optional[int] = None,
        random_state: int = 42,
    ) -> List[DatasetItem]:
        """Rebalances sentiment classes to eliminate the 59% neutral / 12% negative imbalance.

        Args:
            items: Input dataset items.
            strategy: 'balanced' (equal samples across classes) or 'oversample_minority'.
            target_per_class: Target count per class. If None, computes optimal balanced count.
            random_state: Random seed for reproducibility.

        Returns:
            Rebalanced list of DatasetItem objects.
        """
        rng = random.Random(random_state)

        pos = [i for i in items if i.sentiment == RawLabel.POSITIVE]
        neg = [i for i in items if i.sentiment == RawLabel.NEGATIVE]
        neu = [i for i in items if i.sentiment == RawLabel.NEUTRAL]

        logger.info("Class distribution before rebalance -> Positive: %d, Negative: %d, Neutral: %d", len(pos), len(neg), len(neu))

        if not pos or not neg or not neu:
            logger.warning("Cannot rebalance: one or more sentiment classes are empty.")
            return items

        if strategy == "balanced":
            # Target median size across classes or explicit target
            target = target_per_class or int(sorted([len(pos), len(neg), len(neu)])[1])
            balanced_pos = cls._resample(pos, target, rng)
            balanced_neg = cls._resample(neg, target, rng)
            balanced_neu = cls._resample(neu, target, rng)
            result = balanced_pos + balanced_neg + balanced_neu
            rng.shuffle(result)
            logger.info("Class distribution after 'balanced' rebalance -> %d items per class (Total: %d)", target, len(result))
            return result

        elif strategy == "oversample_minority":
            # Oversample negative to match positive
            target = max(len(pos), len(neg))
            balanced_neg = cls._resample(neg, target, rng)
            result = pos + balanced_neg + neu
            rng.shuffle(result)
            return result

        return items

    @classmethod
    def load_combined_dataset(
        cls,
        phrasebank_agreement: float = 0.75,
        include_synthetic_modern: bool = True,
        synthetic_samples: int = 1500,
        extra_csv_paths: Optional[List[Union[str, Path]]] = None,
        rebalance: bool = True,
        random_state: int = 42,
    ) -> List[DatasetItem]:
        """High-level builder: loads and integrates multi-source datasets with class balancing.

        Returns:
            Consolidated, deduplicated, and balanced list of DatasetItem objects.
        """
        combined: List[DatasetItem] = []

        # 1. Load Financial PhraseBank consensus dataset
        try:
            loader = PhraseBankLoader()
            pb_items = loader.load_agreement_txt(min_agreement=phrasebank_agreement)
            combined.extend(pb_items)
            logger.info("Loaded %d items from PhraseBank (agreement >= %.0f%%)", len(pb_items), phrasebank_agreement * 100)
        except Exception as e:
            logger.warning("Could not load PhraseBank: %s", e)

        # 2. Add Synthetic Modern Stock Market Scenarios
        if include_synthetic_modern:
            syn_items = SyntheticStockGenerator.generate_dataset(
                n_samples=synthetic_samples,
                bull_ratio=0.35,
                bear_ratio=0.45,  # Boost negative samples to cure 12% deficit
                neutral_ratio=0.20,
                random_state=random_state,
            )
            combined.extend(syn_items)
            logger.info("Injected %d synthetic modern stock market items.", len(syn_items))

        # 3. Add any extra user-supplied CSV files
        if extra_csv_paths:
            for p in extra_csv_paths:
                try:
                    extra_items = cls.load_csv(p)
                    combined.extend(extra_items)
                except Exception as e:
                    logger.warning("Could not load extra CSV %s: %s", p, e)

        # 4. Deduplicate
        deduped = cls.deduplicate(combined)

        # 5. Class balance
        if rebalance:
            return cls.rebalance_classes(deduped, strategy="balanced", random_state=random_state)

        return deduped

    @staticmethod
    def _resample(items: List[DatasetItem], target_count: int, rng: random.Random) -> List[DatasetItem]:
        if len(items) == target_count:
            return list(items)
        if len(items) > target_count:
            # Undersample without replacement
            return rng.sample(items, target_count)
        # Oversample with replacement
        oversampled = list(items)
        needed = target_count - len(items)
        oversampled.extend(rng.choices(items, k=needed))
        return oversampled

    @classmethod
    def export_csv(cls, items: List[DatasetItem], filepath: Union[str, Path]) -> None:
        """Exports dataset items to a clean CSV file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {"sentence": i.text, "sentiment": i.sentiment.value, "source": i.source_file or ""}
            for i in items
        ]
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info("Exported %d items to %s", len(items), out_path)
