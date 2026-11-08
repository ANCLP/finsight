"""Example script: Train and evaluate the baseline classifier on FinancialPhraseBank."""

from __future__ import annotations

from pathlib import Path
from fin_sentiment import BaselineTrainer, PhraseBankLoader


def main():
    print("=" * 70)
    print(" 🏋️ Training Baseline Financial Sentiment Classifier")
    print("=" * 70)

    # 1. Load dataset (e.g. 75% consensus agreement tier)
    loader = PhraseBankLoader()
    print("\nLoading Financial PhraseBank dataset (>= 75% consensus)...")
    items = loader.load_agreement_txt(min_agreement=0.75)
    print(f"Loaded {len(items)} human-annotated financial sentences.")

    # 2. Run stratified training pipeline
    save_path = Path("checkpoints/baseline_phrasebank.joblib")
    trainer = BaselineTrainer(c_param=2.5)

    print("\nTraining TF-IDF + Balanced Logistic Regression pipeline...")
    report = trainer.train(
        items,
        test_size=0.15,
        val_size=0.15,
        save_path=save_path,
        random_state=42,
    )

    # 3. Print evaluation results
    tm = report["test_metrics"]
    print("\n" + "=" * 70)
    print(" 📊 Holdout Test Set Evaluation Results:")
    print("=" * 70)
    print(f"   • Overall Accuracy:         {tm['accuracy'] * 100:.2f}%")
    print(f"   • Balanced Accuracy:        {tm['balanced_accuracy'] * 100:.2f}%")
    print(f"   • Macro F1-Score:           {tm['macro_f1']:.4f}")
    print(f"   • Weighted F1-Score:        {tm['weighted_f1']:.4f}")
    print(f"   • Directional Accuracy:     {tm['directional_accuracy'] * 100:.2f}% (Bull vs Bear precision)")

    print("\nPer-Class Breakdown:")
    for cls_name, metrics in tm["per_class"].items():
        print(f"   • {cls_name.upper():<8} -> Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")

    print(f"\nModel successfully saved to: {report['saved_model_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
