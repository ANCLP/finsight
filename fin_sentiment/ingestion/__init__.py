"""Ingestion modules for Financial PhraseBank and financial news sources."""

from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.ingestion.text_preprocessor import TextPreprocessor
from fin_sentiment.ingestion.article_parser import ArticleParser

__all__ = [
    "PhraseBankLoader",
    "TextPreprocessor",
    "ArticleParser",
]
