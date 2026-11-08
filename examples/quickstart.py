"""Quickstart example demonstrating fisight capabilities."""

from __future__ import annotations

from fin_sentiment import FinancialNewsSentimentAnalyzer


def main():
    print("=" * 70)
    print(" 🚀 fin-sentiment Quickstart Demo")
    print("=" * 70)

    # 1. Initialize analyzer
    analyzer = FinancialNewsSentimentAnalyzer()

    # 2. Analyze individual headline / sentence
    sample_headline = "Tata Motors consolidated net profit surged 137% beating analyst estimates as JLR margins expanded."
    print(f"\n1. Analyzing Headline:\n   \"{sample_headline}\"")

    score = analyzer.analyze_text(sample_headline)
    print(f"   • Sentiment:       {score.label.value.upper()}")
    print(f"   • Bull/Bear Score: {score.bull_bear_score:+.2f}  (-1.0 = Max Bearish, +1.0 = Max Bullish)")
    print(f"   • Confidence:      {score.confidence * 100:.1f}% ({score.confidence_level.value})")
    print(f"   • Impact Grade:    {score.impact_grade.value}")
    print(f"   • Probabilities:   Bullish: {score.probabilities['bullish']*100:.1f}%, Neutral: {score.probabilities['neutral']*100:.1f}%, Bearish: {score.probabilities['bearish']*100:.1f}%")

    # 3. Analyze complete article with multi-entity attribution
    sample_article = """Tech Earnings Divergence: TCS Beats Expectations While Infosys Faces Margin Squeeze

Tata Consultancy Services ($TCS) reported a 14.8% YoY jump in quarterly net profit, clinching a multi-million dollar banking deal in the UK. Operating margins expanded by 80 basis points.

On the other hand, Infosys ($INFY) missed revenue expectations due to discretionary spending cuts. Operating margin came down to 20.5% from 22.1%, and management announced plans to trim hiring.

Meanwhile, Reliance Industries announced a strategic partnership to expand cloud data centers across India.
"""

    print("\n" + "=" * 70)
    print("2. Analyzing Complete Article with Multi-Stock Attribution:")
    print("=" * 70)

    analysis = analyzer.analyze_article(sample_article)

    print(f"\nHeadline: {analysis.headline}")
    print(f"Overall Verdict: {analysis.overall_sentiment.label.value.upper()} (Bull/Bear Score: {analysis.overall_sentiment.bull_bear_score:+.2f})")
    print(f"Confidence:      {analysis.overall_sentiment.confidence * 100:.1f}%")

    print("\nEntity-Level Sentiment Breakdown:")
    for ticker, entity in analysis.entities.items():
        s = entity.sentiment
        print(f"   • {ticker:<12}: {s.label.value.upper():<8} (Score: {s.bull_bear_score:+.2f}, Confidence: {s.confidence*100:.1f}%, Mentions: {entity.mentions_count})")

    print("\nIdentified Market Catalysts:")
    for cat in analysis.catalysts:
        print(f"   • [{cat.catalyst.value}] '{cat.matched_phrase}' -> {cat.sentiment_bias.value.upper()} (weight: {cat.weight})")

    print("\nExecutive Trading Summary:")
    print(f"   {analysis.summary}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
