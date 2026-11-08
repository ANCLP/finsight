"""Test fixtures and shared helpers for fisight."""

from __future__ import annotations

from pathlib import Path
import pytest

from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.types import DatasetItem, RawLabel


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Returns path to the sentiment_analysis_fin_news root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_dataset_items() -> list[DatasetItem]:
    """Provides a small synthetic set of balanced financial sentences for fast testing."""
    return [
        DatasetItem(
            text="Operating profit rose by 24% to EUR 14.5 million, beating analyst estimates.",
            sentiment=RawLabel.POSITIVE,
            agreement_level=1.0,
            source_file="test_sample",
        ),
        DatasetItem(
            text="Net sales climbed 12.5% to EUR 110 million on strong demand across European markets.",
            sentiment=RawLabel.POSITIVE,
            agreement_level=0.75,
            source_file="test_sample",
        ),
        DatasetItem(
            text="The board has recommended an interim dividend of EUR 0.50 per share.",
            sentiment=RawLabel.POSITIVE,
            agreement_level=1.0,
            source_file="test_sample",
        ),
        DatasetItem(
            text="Operating loss widened to EUR 8.2 million from EUR 2.1 million a year earlier.",
            sentiment=RawLabel.NEGATIVE,
            agreement_level=1.0,
            source_file="test_sample",
        ),
        DatasetItem(
            text="The company announced plans to cut 250 jobs as part of an aggressive restructuring program.",
            sentiment=RawLabel.NEGATIVE,
            agreement_level=0.75,
            source_file="test_sample",
        ),
        DatasetItem(
            text="Operating margin contracted to 3.2% from 6.8% due to higher raw material inflation.",
            sentiment=RawLabel.NEGATIVE,
            agreement_level=1.0,
            source_file="test_sample",
        ),
        DatasetItem(
            text="The company will publish its second-quarter interim financial results on August 15.",
            sentiment=RawLabel.NEUTRAL,
            agreement_level=1.0,
            source_file="test_sample",
        ),
        DatasetItem(
            text="According to the report, the headquarters will remain in Helsinki.",
            sentiment=RawLabel.NEUTRAL,
            agreement_level=1.0,
            source_file="test_sample",
        ),
        DatasetItem(
            text="The shares are listed on the main market of NASDAQ OMX Helsinki Ltd.",
            sentiment=RawLabel.NEUTRAL,
            agreement_level=1.0,
            source_file="test_sample",
        ),
    ]


@pytest.fixture(scope="session")
def real_phrasebank_loader(repo_root: Path) -> PhraseBankLoader:
    """Provides PhraseBankLoader bound to the actual repository dataset directory."""
    return PhraseBankLoader(data_dir=repo_root)
