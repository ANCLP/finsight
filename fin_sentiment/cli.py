"""Command-line interface (CLI) for fin-sentiment.

Allows users to analyze texts, evaluate models, and train classifiers directly from the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fin_sentiment.analyzer import FinancialNewsSentimentAnalyzer
from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.training.baseline_trainer import BaselineTrainer


def main() -> None:
    """CLI entry point for fin-sentiment."""
    parser = argparse.ArgumentParser(
        prog="fin-sentiment",
        description="Production-grade financial news sentiment analysis and Bull/Bear indexer.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command 1: analyze text
    p_analyze = subparsers.add_parser("analyze", help="Analyze financial text or headline")
    p_analyze.add_argument("text", type=str, help="Financial sentence or headline to analyze")
    p_analyze.add_argument("--model-path", type=str, default=None, help="Path to saved model weights")
    p_analyze.add_argument("--json", action="store_true", help="Output results as JSON")

    # Command 2: analyze article from file
    p_article = subparsers.add_parser("analyze-article", help="Analyze a full article from a file")
    p_article.add_argument("filepath", type=str, help="Path to article text or JSON file")
    p_article.add_argument("--ticker", type=str, default=None, help="Default ticker to associate")
    p_article.add_argument("--json", action="store_true", help="Output results as JSON")

    # Command 3: train
    p_train = subparsers.add_parser("train", help="Train a sentiment model on FinancialPhraseBank")
    p_train.add_argument("--agreement", type=float, default=0.75, help="Consensus agreement threshold (0.50, 0.66, 0.75, 1.00)")
    p_train.add_argument("--save-path", type=str, default="checkpoints/baseline_phrasebank.joblib", help="Where to save model")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "analyze":
        analyzer = FinancialNewsSentimentAnalyzer(model_path=args.model_path)
        score = analyzer.analyze_text(args.text)
        if args.json:
            print(json.dumps(score.model_dump(), indent=2))
        else:
            print("\n================ FINANCIAL SENTIMENT ANALYSIS ================")
            print(f"Text:           {args.text}")
            print(f"Sentiment:      {score.label.value.upper()}")
            print(f"Bull/Bear Score: {score.bull_bear_score:+.2f}  (-1.0 = Max Bear, +1.0 = Max Bull)")
            print(f"Confidence:     {score.confidence * 100:.1f}% ({score.confidence_level.value})")
            print(f"Impact Grade:   {score.impact_grade.value}")
            print("Probabilities:  " + ", ".join(f"{k}: {v*100:.1f}%" for k, v in score.probabilities.items()))
            print("==============================================================\n")

    elif args.command == "analyze-article":
        path = Path(args.filepath)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

        content = path.read_text(encoding="utf-8", errors="replace")
        analyzer = FinancialNewsSentimentAnalyzer()
        res = analyzer.analyze_article(content, default_ticker=args.ticker)

        if args.json:
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("\n================ ARTICLE SENTIMENT REPORT ================")
            if res.headline:
                print(f"Headline:        {res.headline}")
            print(f"Overall Verdict: {res.overall_sentiment.label.value.upper()} (Score: {res.overall_sentiment.bull_bear_score:+.2f})")
            print(f"Confidence:      {res.overall_sentiment.confidence * 100:.1f}%")
            print(f"Total Sentences: {len(res.sentences)}")
            if res.entities:
                print("\nIdentified Stock Entities:")
                for sym, ent in res.entities.items():
                    print(f"  • {sym}: {ent.sentiment.label.value.upper()} (Score: {ent.sentiment.bull_bear_score:+.2f}, Mentions: {ent.mentions_count})")
            if res.catalysts:
                print("\nDetected Market Catalysts:")
                for c in res.catalysts:
                    print(f"  • [{c.catalyst.value}] '{c.matched_phrase}' -> {c.sentiment_bias.value.upper()}")
            print("\nSummary:")
            print(f"  {res.summary}")
            print("==========================================================\n")

    elif args.command == "train":
        loader = PhraseBankLoader()
        items = loader.load_agreement_txt(min_agreement=args.agreement)
        print(f"Loaded {len(items)} samples with agreement >= {args.agreement*100:.0f}%.")

        trainer = BaselineTrainer()
        report = trainer.train(items, save_path=args.save_path)
        print("\nTraining complete!")
        print(f"Model saved to: {report['saved_model_path']}")
        print(f"Test Accuracy:  {report['test_metrics']['accuracy']*100:.2f}%")
        print(f"Test Macro F1:  {report['test_metrics']['macro_f1']:.4f}")
        print(f"Directional Acc:{report['test_metrics']['directional_accuracy']*100:.2f}%\n")


if __name__ == "__main__":
    main()
