"""Domain data types, enums, and response models for fin-sentiment.

Follows strict typing and Pydantic v2 schemas for robust serialization,
clean validation, and open-source interoperability.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


class SentimentLabel(str, Enum):
    """Trading-oriented sentiment classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RawLabel(str, Enum):
    """Original annotation labels from FinancialPhraseBank."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

    def to_sentiment_label(self) -> SentimentLabel:
        """Map raw phrase bank label to trading sentiment."""
        mapping = {
            RawLabel.POSITIVE: SentimentLabel.BULLISH,
            RawLabel.NEGATIVE: SentimentLabel.BEARISH,
            RawLabel.NEUTRAL: SentimentLabel.NEUTRAL,
        }
        return mapping[self]


class ImpactGrade(str, Enum):
    """Significance / volatility impact grade of the financial event."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class ConfidenceLevel(str, Enum):
    """Categorical confidence classification."""
    VERY_HIGH = "VERY_HIGH"  # >= 0.85
    HIGH = "HIGH"            # >= 0.70
    MODERATE = "MODERATE"    # >= 0.55
    LOW = "LOW"              # < 0.55


class CatalystType(str, Enum):
    """Identified financial catalysts influencing stock valuation."""
    EARNINGS_BEAT = "EARNINGS_BEAT"
    EARNINGS_MISS = "EARNINGS_MISS"
    REVENUE_GROWTH = "REVENUE_GROWTH"
    REVENUE_DECLINE = "REVENUE_DECLINE"
    MARGIN_EXPANSION = "MARGIN_EXPANSION"
    MARGIN_CONTRACTION = "MARGIN_CONTRACTION"
    DIVIDEND_BUYBACK = "DIVIDEND_BUYBACK"
    DEBT_REDUCTION = "DEBT_REDUCTION"
    DEBT_EXPANSION = "DEBT_EXPANSION"
    REGULATORY_FINE = "REGULATORY_FINE"
    AUDITOR_RESIGNATION = "AUDITOR_RESIGNATION"
    SEC_SEBI_PROBE = "SEC_SEBI_PROBE"
    FILING_DELAY = "FILING_DELAY"
    COUPON_DEFAULT = "COUPON_DEFAULT"
    LAYOFF_RESTRUCTURING = "LAYOFF_RESTRUCTURING"
    M_AND_A_EXPANSION = "M_AND_A_EXPANSION"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    CONTRACT_WIN = "CONTRACT_WIN"
    RATING_UPGRADE = "RATING_UPGRADE"
    RATING_DOWNGRADE = "RATING_DOWNGRADE"
    PRICE_BREAKOUT = "PRICE_BREAKOUT"
    PRICE_BREAKDOWN = "PRICE_BREAKDOWN"


class SentimentScore(BaseModel):
    """Comprehensive sentiment quantification including calibrated probabilities

    and continuous Bull/Bear polarity indexing (-1.00 to +1.00).
    """
    label: SentimentLabel = Field(
        description="Winning sentiment classification: bullish, bearish, or neutral."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Calibrated probability (0.0 to 1.0) of the selected winning class."
    )
    bull_bear_score: float = Field(
        ge=-1.0, le=1.0,
        description="Continuous polarity index from -1.0 (strongly bearish) to +1.0 (strongly bullish)."
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated probability distribution across all 3 sentiment states."
    )
    impact_grade: ImpactGrade = Field(
        default=ImpactGrade.LOW,
        description="Estimated market moving weight based on confidence and score extremity."
    )

    @computed_field
    def confidence_level(self) -> ConfidenceLevel:
        """Categorical representation of confidence score."""
        if self.confidence >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        if self.confidence >= 0.70:
            return ConfidenceLevel.HIGH
        if self.confidence >= 0.55:
            return ConfidenceLevel.MODERATE
        return ConfidenceLevel.LOW

    @classmethod
    def from_probabilities(
        cls,
        probs: Dict[str, float],
        catalyst_boost: float = 0.0,
    ) -> SentimentScore:
        """Constructs a calibrated SentimentScore from probability map.

        Args:
            probs: Dict with keys 'bullish' (or 'positive'), 'bearish' ('negative'), 'neutral'
            catalyst_boost: Optional directional adjustment from identified catalysts (-0.2 to +0.2)
        """
        p_bull = float(probs.get("bullish", probs.get("positive", 0.0)))
        p_bear = float(probs.get("bearish", probs.get("negative", 0.0)))
        p_neut = float(probs.get("neutral", 0.0))

        # Normalize probabilities if sum deviates
        total = p_bull + p_bear + p_neut
        if total > 0:
            p_bull /= total
            p_bear /= total
            p_neut /= total

        # Probability recalibration with catalyst directional bias
        if catalyst_boost > 0:
            shift = min(p_neut * 0.75, catalyst_boost)
            p_bull += shift
            p_neut -= shift
        elif catalyst_boost < 0:
            shift = min(p_neut * 0.75, abs(catalyst_boost))
            p_bear += shift
            p_neut -= shift

        # Continuous polarity metric (-1.0 to +1.0):
        raw_score = p_bull - p_bear
        bull_bear_score = max(-1.0, min(1.0, round(raw_score, 4)))

        # Winning class determination (with polarity thresholding)
        prob_dict = {
            SentimentLabel.BULLISH: p_bull,
            SentimentLabel.BEARISH: p_bear,
            SentimentLabel.NEUTRAL: p_neut,
        }
        if bull_bear_score <= -0.15 and p_bear >= p_bull:
            winning_label = SentimentLabel.BEARISH
            confidence = round(max(p_bear, abs(bull_bear_score)), 4)
        elif bull_bear_score >= 0.15 and p_bull >= p_bear:
            winning_label = SentimentLabel.BULLISH
            confidence = round(max(p_bull, bull_bear_score), 4)
        else:
            winning_label = max(prob_dict, key=prob_dict.get)  # type: ignore
            confidence = round(prob_dict[winning_label], 4)

        # Impact grading based on confidence and polar intensity
        abs_score = abs(bull_bear_score)
        if confidence >= 0.75 and abs_score >= 0.45:
            impact = ImpactGrade.HIGH
        elif confidence >= 0.55 and abs_score >= 0.20:
            impact = ImpactGrade.MODERATE
        else:
            impact = ImpactGrade.LOW

        return cls(
            label=winning_label,
            confidence=confidence,
            bull_bear_score=bull_bear_score,
            probabilities={
                "bullish": round(p_bull, 4),
                "neutral": round(p_neut, 4),
                "bearish": round(p_bear, 4),
            },
            impact_grade=impact,
        )


class CatalystMatch(BaseModel):
    """A detected financial market catalyst in text."""
    catalyst: CatalystType
    matched_phrase: str
    sentiment_bias: SentimentLabel
    weight: float = Field(ge=0.0, le=1.0, default=0.1)


class EntitySentiment(BaseModel):
    """Sentiment evaluation mapped directly to a specific stock entity / ticker."""
    symbol: str = Field(description="Stock symbol / ticker (e.g. RELIANCE, TCS, AAPL)")
    company_name: Optional[str] = None
    mentions_count: int = 1
    sentiment: SentimentScore
    associated_sentences: List[str] = Field(default_factory=list)


class SentenceAnalysis(BaseModel):
    """Granular sentence-level analysis result."""
    index: int
    text: str
    sentiment: SentimentScore
    tickers: List[str] = Field(default_factory=list)
    catalysts: List[CatalystMatch] = Field(default_factory=list)


class ArticleAnalysis(BaseModel):
    """High-level analysis of an entire financial news article."""
    headline: Optional[str] = None
    body: str
    overall_sentiment: SentimentScore
    entities: Dict[str, EntitySentiment] = Field(default_factory=dict)
    sentences: List[SentenceAnalysis] = Field(default_factory=list)
    catalysts: List[CatalystMatch] = Field(default_factory=list)
    summary: str = ""


class DatasetItem(BaseModel):
    """Single sample representation for training and evaluation."""
    text: str
    sentiment: RawLabel
    agreement_level: Optional[float] = Field(
        default=None,
        description="Annotator consensus strength (1.0 = 100%, 0.75 = 75%, 0.66 = 66%, 0.50 = 50%)"
    )
    source_file: Optional[str] = None
