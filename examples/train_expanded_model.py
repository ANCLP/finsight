"""Example script: Train and evaluate baseline model with expanded multi-source stock news."""

from __future__ import annotations

import time
from pathlib import Path

from fin_sentiment.ingestion.dataset_integrator import DatasetIntegrator
from fin_sentiment.models.baseline_model import BaselineSentimentModel
from fin_sentiment.training.baseline_trainer import BaselineTrainer


def main():
    print("=" * 75)
    print(" 🚀 Training Expanded Financial Sentiment Classifier with Modern Stocks")
    print("=" * 75)

    # 1. Integrate Multi-Source Dataset (PhraseBank + Modern Stock News)
    print("\n1. Integrating multi-source dataset (PhraseBank 75% + Modern Equities)...")
    combined_items = DatasetIntegrator.load_combined_dataset(
        phrasebank_agreement=0.75,
        include_synthetic_modern=True,
        synthetic_samples=1800,  # Curated modern US & Indian stock scenarios
        rebalance=False,         # Will be rebalanced inside trainer for pure test sets
        random_state=42,
    )
    print(f"   • Total Integrated Samples: {len(combined_items)}")

    # Class distribution
    pos = sum(1 for i in combined_items if i.sentiment.value == "positive")
    neg = sum(1 for i in combined_items if i.sentiment.value == "negative")
    neu = sum(1 for i in combined_items if i.sentiment.value == "neutral")
    print(f"   • Distribution before train split: Bullish={pos}, Bearish={neg}, Neutral={neu}")

    # 2. Run stratified training pipeline with training-set class rebalancing
    save_path = Path("checkpoints/expanded_stock_model.joblib")
    trainer = BaselineTrainer(c_param=2.5, max_features=25000)

    print("\n2. Training TF-IDF + Balanced Logistic Regression pipeline...")
    t0 = time.perf_counter()
    report = trainer.train(
        combined_items,
        test_size=0.15,
        val_size=0.15,
        save_path=save_path,
        rebalance_train_classes=True,  # Balances training set to cure neutral/negative imbalance
        random_state=42,
    )
    train_time = time.perf_counter() - t0
    print(f"   • Training completed in {train_time:.2f} seconds on CPU!")

    # 3. Print evaluation results
    tm = report["test_metrics"]
    print("\n" + "=" * 75)
    print(" 📊 Pure Holdout Test Set Evaluation Results:")
    print("=" * 75)
    print(f"   • Overall Accuracy:         {tm['accuracy'] * 100:.2f}%")
    print(f"   • Balanced Accuracy:        {tm['balanced_accuracy'] * 100:.2f}%")
    print(f"   • Macro F1-Score:           {tm['macro_f1']:.4f}")
    print(f"   • Weighted F1-Score:        {tm['weighted_f1']:.4f}")
    print(f"   • Directional Accuracy:     {tm['directional_accuracy'] * 100:.2f}% (Bull vs Bear precision)")

    print("\nPer-Class Metrics:")
    for cls_name, metrics in tm["per_class"].items():
        print(f"   • {cls_name.upper():<8} -> Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")

    # 4. Latency Benchmark: Verify < 1ms per sample on CPU
    print("\n" + "=" * 75)
    print(" ⚡ Latency & Resource Footprint Verification:")
    print("=" * 75)

    test_model = BaselineSentimentModel(model_path=save_path)
    sample_texts = [
        "Nvidia quarterly data center revenue surged 427% driven by demand for AI accelerators.",
        "Regulatory authorities issued a formal show-cause notice to the company regarding debt defaults.",
        "The board of directors declared an interim cash dividend of $0.50 per share.",
        "Operating margins contracted by 180 basis points due to severe supply chain cost inflation.",
        "The company announced it will publish its quarterly earnings report on November 15.",
    ] * 200  # 1,000 sentences

    t_start = time.perf_counter()
    scores = test_model.predict_batch(sample_texts)
    total_time = time.perf_counter() - t_start
    per_sample_ms = (total_time / len(sample_texts)) * 1000

    checkpoint_size_mb = save_path.stat().st_size / (1024 * 1024)

    print(f"   • Sentences Evaluated:      {len(sample_texts)}")
    print(f"   • Total Batch Inference:    {total_time * 1000:.2f} ms")
    print(f"   • Latency Per Sentence:     {per_sample_ms:.3f} ms (Target: < 1.0 ms)")
    print(f"   • Serialized Model Size:    {checkpoint_size_mb:.2f} MB (Target: < 2.0 MB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
