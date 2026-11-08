"""Example script: Head-to-head benchmark comparing Legacy PhraseBank vs. Expanded Modern Stock Model."""

from __future__ import annotations

import time
from fin_sentiment.ingestion.dataset_integrator import DatasetIntegrator
from fin_sentiment.ingestion.phrasebank_loader import PhraseBankLoader
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.training.baseline_trainer import BaselineTrainer


def main():
    print("=" * 80)
    print(" 🔬 SIDE-BY-SIDE MODEL BENCHMARK: LEGACY vs. EXPANDED MODERN STOCK MODEL")
    print("=" * 80)

    # 1. Train Legacy Baseline (Financial PhraseBank Only)
    print("\n[1/3] Loading & Training Legacy Baseline Model (PhraseBank 75% Agreement)...")
    loader = PhraseBankLoader()
    pb_items = loader.load_agreement_txt(min_agreement=0.75)

    legacy_trainer = BaselineTrainer(c_param=2.5, max_features=None)
    t0 = time.perf_counter()
    legacy_report = legacy_trainer.train(
        pb_items,
        test_size=0.15,
        val_size=0.15,
        rebalance_train_classes=False,  # Unbalanced (12.5% negative deficit)
        random_state=42,
    )
    legacy_train_time = time.perf_counter() - t0
    legacy_model = legacy_trainer.model
    print(f"      Legacy training completed in {legacy_train_time:.2f}s (Total samples: {len(pb_items)})")

    # 2. Train Expanded Baseline (Multi-Source + Modern Stocks + Balanced)
    print("\n[2/3] Integrating & Training Expanded Model (PhraseBank + Modern Equities + Balanced)...")
    combined_items = DatasetIntegrator.load_combined_dataset(
        phrasebank_agreement=0.75,
        include_synthetic_modern=True,
        synthetic_samples=1800,
        rebalance=False,
        random_state=42,
    )

    expanded_trainer = BaselineTrainer(c_param=2.5, max_features=25000)
    t0 = time.perf_counter()
    expanded_report = expanded_trainer.train(
        combined_items,
        test_size=0.15,
        val_size=0.15,
        save_path="checkpoints/expanded_stock_model.joblib",
        rebalance_train_classes=True,  # Balanced training split (curing 12% negative deficit)
        random_state=42,
    )
    expanded_train_time = time.perf_counter() - t0
    expanded_model = expanded_trainer.model
    print(f"      Expanded training completed in {expanded_train_time:.2f}s (Total samples: {len(combined_items)})")

    # 3. Benchmark Metrics Comparison on Holdout Test Sets
    ltm = legacy_report["test_metrics"]
    etm = expanded_report["test_metrics"]

    print("\n" + "=" * 80)
    print(" 📊 HOLDOUT EVALUATION METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Legacy Baseline':<20} | {'Expanded Model':<20}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<35} | {ltm['accuracy']*100:>18.2f}% | {etm['accuracy']*100:>18.2f}%")
    print(f"{'Balanced Accuracy':<35} | {ltm['balanced_accuracy']*100:>18.2f}% | {etm['balanced_accuracy']*100:>18.2f}%")
    print(f"{'Macro F1-Score':<35} | {ltm['macro_f1']:>19.4f} | {etm['macro_f1']:>19.4f}")
    print(f"{'Weighted F1-Score':<35} | {ltm['weighted_f1']:>19.4f} | {etm['weighted_f1']:>19.4f}")
    print(f"{'Directional Bull/Bear Accuracy':<35} | {ltm['directional_accuracy']*100:>18.2f}% | {etm['directional_accuracy']*100:>18.2f}%")
    print("-" * 80)
    print("Per-Class F1 Scores:")
    for cls in ["bearish", "neutral", "bullish"]:
        l_f1 = ltm["per_class"].get(cls, {}).get("f1", 0.0)
        e_f1 = etm["per_class"].get(cls, {}).get("f1", 0.0)
        print(f"  • {cls.upper():<33} | {l_f1:>19.3f} | {e_f1:>19.3f}")

    # 4. Real-World Stock Headline Test Bench
    test_cases = [
        ("Nvidia quarterly data center revenue surged 427% driven by demand for AI accelerators.", "bullish"),
        ("Tesla slashed vehicle prices by 15% as operating margins tumbled to multi-year lows.", "bearish"),
        ("Statutory auditor resigned citing material accounting irregularities in revenue recognition.", "bearish"),
        ("Infosys warned of slowing enterprise digital spending and cut its constant currency guidance.", "bearish"),
        ("SEBI issued a formal show-cause notice to the company regarding undisclosed inter-corporate borrowings.", "bearish"),
        ("Tata Consultancy Services clinched a $1.2B banking deal with British insurer.", "bullish"),
        ("The company announced plans to eliminate 10,000 corporate positions to lower fixed costs.", "bearish"),
        ("Apple reported record Services revenue of $24 billion, beating Street consensus views.", "bullish"),
        ("Zomato turned profitable for the first time as quick commerce contribution margins expanded.", "bullish"),
        ("Operating profit decreased by 18% as raw material costs rose.", "bearish"),
        ("Basware net sales increased by 11.2% in the first quarter of 2012.", "bullish"),
        ("The company will publish its second-quarter interim financial report on August 15.", "neutral"),
    ]

    print("\n" + "=" * 80)
    print(" 🎯 HEAD-TO-HEAD PREDICTIONS ON REAL-WORLD FINANCIAL HEADLINES")
    print("=" * 80)

    legacy_matches = 0
    expanded_matches = 0

    for idx, (headline, expected) in enumerate(test_cases, 1):
        leg_score = legacy_model.predict(headline)
        exp_score = expanded_model.predict(headline)

        leg_correct = leg_score.label.value == expected
        exp_correct = exp_score.label.value == expected

        if leg_correct:
            legacy_matches += 1
        if exp_correct:
            expanded_matches += 1

        print(f"\nHeadline #{idx}: \"{headline}\"")
        print(f"Expected Sentiment: {expected.upper()}")
        print(f"  • Legacy Baseline: {leg_score.label.value.upper():<8} (Score: {leg_score.bull_bear_score:+.2f}, Conf: {leg_score.confidence*100:.1f}%) -> {'[CORRECT]' if leg_correct else '[MISSED]'}")
        print(f"  • Expanded Model:  {exp_score.label.value.upper():<8} (Score: {exp_score.bull_bear_score:+.2f}, Conf: {exp_score.confidence*100:.1f}%) -> {'[CORRECT]' if exp_correct else '[MISSED]'}")

    print("\n" + "=" * 80)
    print(" 🏆 HEAD-TO-HEAD TEST BENCH SUMMARY")
    print("=" * 80)
    print(f"Legacy Baseline Correct: {legacy_matches} / {len(test_cases)} ({legacy_matches / len(test_cases) * 100:.1f}%)")
    print(f"Expanded Model Correct:  {expanded_matches} / {len(test_cases)} ({expanded_matches / len(test_cases) * 100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
