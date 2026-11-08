"""fin-sentiment: Production-grade financial news sentiment analysis and Bull/Bear indexer.

Open-source Python library for trading desks, quantitative finance researchers,
and individual algorithmic traders to extract actionable market sentiment from news articles.
"""

from fin_sentiment.analyzer import FinancialNewsSentimentAnalyzer
from fin_sentiment.entity.financial_catalysts import CatalystExtractor
from fin_sentiment.entity.ticker_extractor import TickerExtractor
from fin_sentiment.ingestion.article_parser import ArticleParser
from fin_sentiment.ingestion.discourse_deconstructor import DiscourseDeconstructor
from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.ingestion.text_preprocessor import TextPreprocessor
from fin_sentiment.models.base import BaseSentimentModel
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.models.finbert_model import FinBertSentimentModel
from fin_sentiment.models.llm_arbiter import GeminiFinancialArbiter
from fin_sentiment.models.onnx_model import OnnxSentimentModel
from fin_sentiment.training.baseline_trainer import BaselineTrainer
from fin_sentiment.training.metrics import FinancialMetrics
from fin_sentiment.training.transformer_trainer import TransformerTrainer
from fin_sentiment.types import (
    ArticleAnalysis,
    CatalystMatch,
    CatalystType,
    ConfidenceLevel,
    DatasetItem,
    EntitySentiment,
    ImpactGrade,
    RawLabel,
    SentenceAnalysis,
    SentimentLabel,
    SentimentScore,
)

__version__ = "0.1.0"
__all__ = [
    # Top-level Analyzer
    "FinancialNewsSentimentAnalyzer",
    # Types & Models
    "SentimentScore",
    "SentimentLabel",
    "RawLabel",
    "ImpactGrade",
    "ConfidenceLevel",
    "CatalystType",
    "CatalystMatch",
    "EntitySentiment",
    "SentenceAnalysis",
    "ArticleAnalysis",
    "DatasetItem",
    # Models
    "BaseSentimentModel",
    "BaselineSentimentModel",
    "FinBertSentimentModel",
    "OnnxSentimentModel",
    "GeminiFinancialArbiter",
    # Ingestion & Preprocessing
    "PhraseBankLoader",
    "TextPreprocessor",
    "ArticleParser",
    "DiscourseDeconstructor",
    # Entities & Catalysts
    "TickerExtractor",
    "CatalystExtractor",
    # Training & Metrics
    "FinancialMetrics",
    "BaselineTrainer",
    "TransformerTrainer",
]
