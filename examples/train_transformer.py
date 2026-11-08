"""Template script: Fine-tune a Transformer model (FinBERT / RoBERTa) on FinancialPhraseBank."""

from __future__ import annotations

from fin_sentiment import PhraseBankLoader, TransformerTrainer


def main():
    print("=" * 70)
    print(" 🤖 Fine-Tuning FinBERT Transformer on Financial PhraseBank")
    print("=" * 70)

    # 1. Load dataset (100% agreement tier for pristine training signal)
    loader = PhraseBankLoader()
    print("\nLoading gold-standard Financial PhraseBank dataset (100% agreement)...")
    items = loader.load_agreement_txt(min_agreement=1.00)
    print(f"Loaded {len(items)} human-annotated sentences.")

    # 2. Configure TransformerTrainer
    trainer = TransformerTrainer(
        base_model_name="ProsusAI/finbert",
        output_dir="checkpoints/finbert_phrasebank",
        learning_rate=2e-5,
        epochs=3,
        batch_size=16,
        max_length=128,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
    )

    print("\nStarting fine-tuning workflow (requires GPU for optimal speed)...")
    try:
        report = trainer.train(items)
        print("\nFine-tuning completed successfully!")
        print(f"Model saved to: {report['checkpoint_dir']}")
        print(f"Test metrics: {report['test_metrics']}")
    except Exception as e:
        print(f"\nNotice: Transformer training requires torch & transformers with appropriate compute/internet access: {e}")
        print("For zero-setup CPU environments, use BaselineTrainer (TF-IDF + Logistic Regression).")


if __name__ == "__main__":
    main()
