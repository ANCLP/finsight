"""Tests for dataset loading from all-data.csv and FinancialPhraseBank txt files."""

from __future__ import annotations

import pytest
from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.types import RawLabel


def test_load_all_data_csv(real_phrasebank_loader: PhraseBankLoader):
    items = real_phrasebank_loader.load_csv()
    assert len(items) > 4000
    assert all(isinstance(i.text, str) and len(i.text) > 0 for i in items)
    assert all(isinstance(i.sentiment, RawLabel) for i in items)

    # Verify class balance presence
    sentiments = {i.sentiment for i in items}
    assert RawLabel.POSITIVE in sentiments
    assert RawLabel.NEGATIVE in sentiments
    assert RawLabel.NEUTRAL in sentiments


def test_load_agreement_txt_all_agree(real_phrasebank_loader: PhraseBankLoader):
    # Sentences_AllAgree.txt has 2264 samples with 100% human annotator agreement
    items = real_phrasebank_loader.load_agreement_txt(min_agreement=1.00)
    assert len(items) == 2264
    assert all(i.agreement_level == 1.00 for i in items)


def test_load_agreement_txt_75_agree(real_phrasebank_loader: PhraseBankLoader):
    # Sentences_75Agree.txt has 3453 samples
    items = real_phrasebank_loader.load_agreement_txt(min_agreement=0.75)
    assert len(items) == 3453
    assert all(i.agreement_level == 0.75 for i in items)


def test_load_dataset_auto(real_phrasebank_loader: PhraseBankLoader):
    items = real_phrasebank_loader.load_dataset()
    assert len(items) >= 2264


def test_to_dataframe(real_phrasebank_loader: PhraseBankLoader):
    items = real_phrasebank_loader.load_agreement_txt(min_agreement=1.00)
    df = real_phrasebank_loader.to_dataframe(items[:50])
    assert len(df) == 50
    assert list(df.columns) == ["text", "raw_sentiment", "sentiment", "agreement_level", "source_file"]


def test_stratified_splits(real_phrasebank_loader: PhraseBankLoader):
    items = real_phrasebank_loader.load_agreement_txt(min_agreement=1.00)
    subset = items[:1000]
    train, val, test = real_phrasebank_loader.create_splits(subset, test_size=0.15, val_size=0.15, random_state=42)

    assert len(train) + len(val) + len(test) == len(subset)
    # Check that each split has non-zero positive, negative, neutral
    for split in (train, val, test):
        s_types = {i.sentiment for i in split}
        assert len(s_types) == 3
