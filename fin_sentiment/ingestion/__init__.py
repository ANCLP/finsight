"""Ingestion modules for Financial PhraseBank and financial news sources."""

from fin_sentiment.ingestion.article_parser import ArticleParser
from fin_sentiment.ingestion.dataset_integrator import DatasetIntegrator
from fin_sentiment.ingestion.discourse_deconstructor import DiscourseDeconstructor
from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.ingestion.synthetic_stock_generator import SyntheticStockGenerator
from fin_sentiment.ingestion.text_preprocessor import TextPreprocessor

__all__ = [
    "PhraseBankLoader",
    "TextPreprocessor",
    "ArticleParser",
    "DiscourseDeconstructor",
    "DatasetIntegrator",
    "SyntheticStockGenerator",
]
