"""Main user-facing facade: FinancialNewsSentimentAnalyzer.

Provides a unified, high-level API to analyze financial news headlines, articles,
social sentiment, and earnings reports with ticker-level attribution and Bull/Bear scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from fin_sentiment.entity.financial_catalysts import CatalystExtractor
from fin_sentiment.entity.ticker_extractor import TickerExtractor
from fin_sentiment.ingestion.article_parser import ArticleParser
from fin_sentiment.ingestion.discourse_deconstructor import DiscourseDeconstructor
from fin_sentiment.models.base import BaseSentimentModel
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.models.finbert_model import FinBertSentimentModel
from fin_sentiment.models.llm_arbiter import GeminiFinancialArbiter
from fin_sentiment.models.onnx_model import OnnxSentimentModel
from fin_sentiment.types import (
    ArticleAnalysis,
    CatalystMatch,
    EntitySentiment,
    SentenceAnalysis,
    SentimentLabel,
    SentimentScore,
)

logger = logging.getLogger(__name__)


class FinancialNewsSentimentAnalyzer:
    """The central sentiment analysis engine for financial news and market intelligence."""

    def __init__(
        self,
        model: Optional[BaseSentimentModel] = None,
        model_backend: str = "baseline",
        model_path: Optional[Union[str, Path]] = None,
        auto_train_baseline_if_missing: bool = True,
        gemini_api_key: Optional[str] = None,
    ):
        """Initializes the analyzer.

        Args:
            model: Custom pre-instantiated BaseSentimentModel backend.
            model_backend: 'baseline' (fast TF-IDF + Logistic Regression), 'onnx' (Quantized INT8), or 'finbert' (Deep Transformer).
            model_path: Path to serialized model weights / checkpoint directory.
            auto_train_baseline_if_missing: If True and baseline model is not trained/saved,
                                            automatically trains on local FinancialPhraseBank data.
            gemini_api_key: Optional Google AI Studio API key for serverless LLM arbitration.
        """
        self.arbiter = GeminiFinancialArbiter(api_key=gemini_api_key)

        if model is not None:
            self.model = model
        elif model_backend == "onnx":
            self.model = OnnxSentimentModel(model_path=model_path)
            if not self.model.is_ready():
                logger.info("ONNX model not ready at %s; falling back to baseline classifier.", model_path)
                self.model = BaselineSentimentModel()
                if not self.model.is_ready() and auto_train_baseline_if_missing:
                    self._auto_fit_baseline()
        elif model_backend == "finbert":
            self.model = FinBertSentimentModel(
                model_name_or_path=str(model_path) if model_path else "ProsusAI/finbert"
            )
        else:
            # Baseline classifier
            self.model = BaselineSentimentModel(model_path=model_path)
            if not self.model.is_ready() and auto_train_baseline_if_missing:
                self._auto_fit_baseline()

    def _auto_fit_baseline(self) -> None:
        """Automatically fits the baseline model on the local FinancialPhraseBank dataset."""
        try:
            from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader

            loader = PhraseBankLoader()
            items = loader.load_dataset()
            if items and isinstance(self.model, BaselineSentimentModel):
                texts = [i.text for i in items]
                labels = [i.sentiment.value for i in items]
                self.model.fit(texts, labels)
                logger.info("Auto-fitted baseline model on %d phrasebank sentences.", len(items))
        except Exception as e:
            logger.warning("Could not auto-fit baseline model: %s", e)

    def analyze_text(
        self,
        text: str,
        associated_ticker: Optional[str] = None,
    ) -> SentimentScore:
        """Analyzes a single financial sentence or headline.

        Employs the 3-Pillar Hybrid Pipeline:
        1. Free-Tier Gemini 1.5 Flash Arbiter for complex multi-entity, consensus, or regulatory red flags.
        2. Discourse & Syntactic Clause Deconstructor for subordinate contrastives, scare quotes, and metric pairing.
        3. Quantized ONNX / Calibrated Baseline with directional catalyst detection.

        Args:
            text: Financial sentence or headline (e.g. "TCS profit surged 14% beating estimates")
            associated_ticker: Optional ticker symbol for context

        Returns:
            Calibrated SentimentScore with Bull/Bear rating, confidence, and probability map.
        """
        # 1. Serverless Free-Tier LLM Arbiter for Red Flag Scenarios (Pillar 3)
        if self.arbiter and self.arbiter.is_available():
            if self.arbiter.should_arbitrate(text, baseline_conf=0.5):
                arb_res = self.arbiter.analyze(text, target_ticker=associated_ticker)
                if arb_res:
                    s_label = SentimentLabel(arb_res["label"])
                    score_val = arb_res["bull_bear_score"]
                    conf = arb_res["confidence"]
                    p_bull = round(max(0.01, min(0.98, 0.5 + score_val / 2.0)), 4)
                    p_bear = round(max(0.01, min(0.98, 0.5 - score_val / 2.0)), 4)
                    p_neut = round(max(0.01, 1.0 - abs(score_val)), 4)
                    tot = p_bull + p_bear + p_neut
                    prob_map = {
                        "bullish": round(p_bull / tot, 4),
                        "bearish": round(p_bear / tot, 4),
                        "neutral": round(p_neut / tot, 4),
                    }
                    return SentimentScore(
                        label=s_label,
                        confidence=conf,
                        bull_bear_score=score_val,
                        probabilities=prob_map,
                    )

        # 2. Discourse Deconstruction & Linguistic Parsing (Pillar 1)
        decon = DiscourseDeconstructor.deconstruct(text)

        # Directional metric pairing override (e.g. "debt expanded" -> Bearish)
        if decon.directional_metric_override is not None:
            override_val = decon.directional_metric_override
            s_label = SentimentLabel.BULLISH if override_val > 0 else SentimentLabel.BEARISH
            p_bull = 0.80 if override_val > 0 else 0.05
            p_bear = 0.80 if override_val < 0 else 0.05
            p_neut = 0.15
            tot = p_bull + p_bear + p_neut
            return SentimentScore(
                label=s_label,
                confidence=0.82,
                bull_bear_score=override_val,
                probabilities={
                    "bullish": round(p_bull / tot, 4),
                    "bearish": round(p_bear / tot, 4),
                    "neutral": round(p_neut / tot, 4),
                },
            )

        # Contrastive clause processing ("Although A, B")
        if decon.is_contrastive and decon.concession_clause:
            op_score = self.model.predict(decon.operative_clause)
            conc_score = self.model.predict(decon.concession_clause)

            # Evaluate catalysts independently for operative vs concession clauses
            op_cats = CatalystExtractor.detect_catalysts(decon.operative_clause)
            conc_cats = CatalystExtractor.detect_catalysts(decon.concession_clause)
            op_boost = CatalystExtractor.compute_catalyst_boost(op_cats)
            conc_boost = CatalystExtractor.compute_catalyst_boost(conc_cats)
            clause_catalyst_boost = (op_boost * decon.operative_weight) + (conc_boost * decon.concession_weight)

            blended_probs = {
                k: round(
                    decon.operative_weight * op_score.probabilities.get(k, 0.33)
                    + decon.concession_weight * conc_score.probabilities.get(k, 0.33),
                    4,
                )
                for k in ["bullish", "bearish", "neutral"]
            }
            return SentimentScore.from_probabilities(blended_probs, catalyst_boost=clause_catalyst_boost)
        else:
            base_score = self.model.predict(text)

        # Scare-quote superlative attenuation (e.g. "'stellar' turnaround")
        if decon.has_scare_quotes:
            p = dict(base_score.probabilities)
            p["bullish"] = p.get("bullish", 0.0) * 0.5
            p["neutral"] = p.get("neutral", 0.0) + (base_score.probabilities.get("bullish", 0.0) * 0.5)
            base_score = SentimentScore.from_probabilities(p)

        # Enhance with catalyst detection (unless negated by denial)
        catalysts = CatalystExtractor.detect_catalysts(text)
        catalyst_boost = (
            0.0 if decon.is_rumor_denial else CatalystExtractor.compute_catalyst_boost(catalysts)
        )

        if catalyst_boost != 0.0:
            return SentimentScore.from_probabilities(
                base_score.probabilities,
                catalyst_boost=catalyst_boost,
            )

        return base_score

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Classifies sentiment across a list of financial texts with high throughput."""
        if not texts:
            return []

        return [self.analyze_text(t) for t in texts]

    def analyze_article(
        self,
        source: Union[str, Dict],
        headline: Optional[str] = None,
        default_ticker: Optional[str] = None,
    ) -> ArticleAnalysis:
        """Performs a deep structured analysis of a complete financial news article.

        Extracts:
        - Overall article sentiment & Bull/Bear rating
        - Identified stock tickers & company entities
        - Entity-level sentiment (e.g. sentiment for $TCS vs $INFY)
        - Sentence-level granular breakdowns
        - Market catalysts (earnings, margins, buybacks, layoffs, fines)
        - Executive trading summary
        """
        parsed_hl, body, sentences = ArticleParser.parse(source, headline=headline)

        if not sentences:
            # Fallback for empty text
            empty_score = SentimentScore(
                label=SentimentLabel.NEUTRAL,
                confidence=0.50,
                bull_bear_score=0.0,
                probabilities={"bullish": 0.25, "neutral": 0.50, "bearish": 0.25},
            )
            return ArticleAnalysis(
                headline=parsed_hl,
                body=body,
                overall_sentiment=empty_score,
                summary="Empty or unparseable article.",
            )

        # 1. Analyze each sentence
        sentence_scores = self.analyze_batch(sentences)
        sentence_analyses: List[SentenceAnalysis] = []
        all_catalysts: List[CatalystMatch] = []
        ticker_sentence_map: Dict[str, List[Tuple[str, SentimentScore]]] = {}

        for idx, (sent_text, score) in enumerate(zip(sentences, sentence_scores)):
            sent_tickers = TickerExtractor.extract_tickers(sent_text)
            if not sent_tickers and default_ticker:
                sent_tickers = [default_ticker]

            sent_catalysts = CatalystExtractor.detect_catalysts(sent_text)
            all_catalysts.extend(sent_catalysts)

            sentence_analyses.append(
                SentenceAnalysis(
                    index=idx + 1,
                    text=sent_text,
                    sentiment=score,
                    tickers=sent_tickers,
                    catalysts=sent_catalysts,
                )
            )

            for sym in sent_tickers:
                if sym not in ticker_sentence_map:
                    ticker_sentence_map[sym] = []
                ticker_sentence_map[sym].append((sent_text, score))

        # 2. Compute aggregate overall sentiment
        # Weight headline higher (weight=2.0) if present
        total_bull = 0.0
        total_bear = 0.0
        total_neut = 0.0
        total_weight = 0.0

        for sa in sentence_analyses:
            weight = 2.0 if (parsed_hl and sa.text.startswith(parsed_hl[:30])) else 1.0
            probs = sa.sentiment.probabilities
            total_bull += probs.get("bullish", 0.0) * weight
            total_bear += probs.get("bearish", 0.0) * weight
            total_neut += probs.get("neutral", 0.0) * weight
            total_weight += weight

        avg_probs = {
            "bullish": total_bull / total_weight,
            "bearish": total_bear / total_weight,
            "neutral": total_neut / total_weight,
        }
        overall_catalyst_boost = CatalystExtractor.compute_catalyst_boost(all_catalysts)
        overall_sentiment = SentimentScore.from_probabilities(
            avg_probs,
            catalyst_boost=overall_catalyst_boost,
        )

        # 3. Compute per-entity sentiment
        entities: Dict[str, EntitySentiment] = {}
        for sym, s_list in ticker_sentence_map.items():
            sym_bull = sum(s[1].probabilities.get("bullish", 0.0) for s in s_list) / len(s_list)
            sym_bear = sum(s[1].probabilities.get("bearish", 0.0) for s in s_list) / len(s_list)
            sym_neut = sum(s[1].probabilities.get("neutral", 0.0) for s in s_list) / len(s_list)

            sym_catalysts = []
            for s in s_list:
                sym_catalysts.extend(CatalystExtractor.detect_catalysts(s[0]))

            sym_boost = CatalystExtractor.compute_catalyst_boost(sym_catalysts)
            sym_score = SentimentScore.from_probabilities(
                {"bullish": sym_bull, "bearish": sym_bear, "neutral": sym_neut},
                catalyst_boost=sym_boost,
            )

            entities[sym] = EntitySentiment(
                symbol=sym,
                mentions_count=len(s_list),
                sentiment=sym_score,
                associated_sentences=[s[0] for s in s_list],
            )

        # 4. Generate summary narrative
        summary_parts = [
            f"Overall Verdict: {overall_sentiment.label.value.upper()} (Bull/Bear Score: {overall_sentiment.bull_bear_score:+.2f}, Confidence: {overall_sentiment.confidence * 100:.1f}%)."
        ]
        if entities:
            entity_summaries = [
                f"{k}: {v.sentiment.label.value.upper()} ({v.sentiment.bull_bear_score:+.2f})"
                for k, v in entities.items()
            ]
            summary_parts.append(f"Entity Attribution: {', '.join(entity_summaries)}.")
        if all_catalysts:
            cat_names = sorted(set(c.catalyst.value for c in all_catalysts))
            summary_parts.append(f"Key Catalysts: {', '.join(cat_names)}.")

        summary_text = " ".join(summary_parts)

        return ArticleAnalysis(
            headline=parsed_hl,
            body=body,
            overall_sentiment=overall_sentiment,
            entities=entities,
            sentences=sentence_analyses,
            catalysts=all_catalysts,
            summary=summary_text,
        )
