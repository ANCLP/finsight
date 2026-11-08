# 📈 fisight

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: passing](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)]()
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Production-grade, open-source financial news sentiment analysis, continuous Bull/Bear polarity indexing, and stock entity attribution engine.**

---

## 🌟 Key Features

- **🎯 Continuous Bull/Bear Polarity Score**: Outputs both discrete classifications (`BULLISH`, `BEARISH`, `NEUTRAL`) and a continuous polarity index from **`-1.00` (Max Bearish)** to **`+1.00` (Max Bullish)**, with calibrated probabilities and impact grading (`HIGH`, `MODERATE`, `LOW`).
- **🏢 Multi-Entity Ticker Attribution**: Analyzes complex financial articles mentioning multiple stocks (e.g. *"TCS profit beat estimates while Infosys faced margin pressure"*) and attributes separate sentiment ratings to each stock individually (`$TCS`, `$INFY`, `$AAPL`, `$RELIANCE`).
- **⚡ Dual Engine Architecture**:
  - **Lightweight Baseline (Production)**: TF-IDF + Calibrated Balanced Logistic Regression. Trains in **< 2 seconds**, runs in **< 1ms per sample on CPU**, and achieves **85.14% holdout accuracy**. Zero heavy dependencies required.
  - **Deep Transformer (FinBERT)**: Built-in adapter for `ProsusAI/finbert` and local fine-tuned HuggingFace / PyTorch checkpoints with GPU acceleration.
- **🔍 Market Catalyst Detection**: Automatically detects 16+ market-moving catalysts (earnings surprises, margin changes, dividend announcements, share buybacks, regulatory penalties, debt restructuring, contract wins).
- **📂 Robust Dataset Ingestion**:
  - Full support for `all-data.csv` (CSV format).
  - Native support for Aalto University's `@`-delimited `Sentences_*Agree.txt` consensus files (100%, 75%, 66%, 50% annotator agreement tiers).
  - Multi-encoding fallback (`latin-1`, `utf-8`, `iso-8859-1`).
  - Automatic stratified train/validation/test split generation.
- **🖥️ Command Line Interface (CLI)**: Out-of-the-box CLI to analyze headlines, parse full articles, and train custom models directly from your terminal.

---

## 📦 Architecture

```
fin_sentiment/
├── types.py                  # Domain types: SentimentScore, EntitySentiment, ArticleAnalysis
├── ingestion/                # Ingestion & preprocessing
│   ├── phrasebank_loader.py  # Loader for all-data.csv and Sentences_*Agree.txt
│   ├── text_preprocessor.py  # Financial decimal/suffix-aware tokenizer & segmenter
│   └── article_parser.py     # Multi-format news article & feed parser
├── entity/                   # Entity & catalyst intelligence
│   ├── ticker_extractor.py   # Cashtags ($TCS), exchange tickers, company names
│   └── financial_catalysts.py# Earnings, margins, layoffs, dividend catalysts
├── models/                   # Classification backends
│   ├── base.py               # Abstract BaseSentimentModel protocol
│   ├── baseline_model.py     # Fast TF-IDF + LogisticRegression model (85%+ acc)
│   └── finbert_model.py      # HuggingFace FinBERT PyTorch adapter
├── training/                 # Model training pipelines
│   ├── baseline_trainer.py   # Automated stratified baseline training & evaluation
│   ├── transformer_trainer.py# Template for fine-tuning FinBERT on custom datasets
│   └── metrics.py            # Financial accuracy, F1, and Bull/Bear directional metrics
├── analyzer.py               # High-level facade: FinancialNewsSentimentAnalyzer
└── cli.py                    # Command-line interface
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/finsentiment/fin-sentiment.git
cd fin-sentiment

# Install core dependencies (CPU baseline)
pip install -e .

# (Optional) Install Deep Learning Transformer support
pip install -e ".[torch]"
```

### 2. Single Headline Analysis

```python
from fin_sentiment import FinancialNewsSentimentAnalyzer

analyzer = FinancialNewsSentimentAnalyzer()
score = analyzer.analyze_text("Tata Motors net profit surged 137% beating analyst estimates as JLR margins expanded.")

print(score.label)            # SentimentLabel.BULLISH
print(score.bull_bear_score)  # +0.60  (-1.0 to +1.0)
print(score.confidence)       # 0.561  (56.1%)
print(score.probabilities)    # {'bullish': 0.561, 'neutral': 0.317, 'bearish': 0.122}
print(score.impact_grade)     # ImpactGrade.MODERATE
```

### 3. Full Article Analysis with Multi-Stock Attribution

```python
from fin_sentiment import FinancialNewsSentimentAnalyzer

analyzer = FinancialNewsSentimentAnalyzer()

article = """
Tech Earnings Divergence: TCS Beats Expectations While Infosys Faces Margin Squeeze

Tata Consultancy Services ($TCS) reported a 14.8% jump in quarterly net profit, clinching a multi-million dollar banking deal. Margins expanded by 80 bps.

On the other hand, Infosys ($INFY) missed revenue expectations due to discretionary spending cuts. Operating margin came down to 20.5% and management announced plans to trim hiring.
"""

result = analyzer.analyze_article(article)

# Overall verdict
print(f"Overall: {result.overall_sentiment.label.value} (Score: {result.overall_sentiment.bull_bear_score:+.2f})")

# Per-stock attribution
for ticker, entity in result.entities.items():
    print(f"{ticker}: {entity.sentiment.label.value} (Score: {entity.sentiment.bull_bear_score:+.2f})")
# Output:
# TCS:  BULLISH (+0.31)
# INFY: BEARISH (-0.11)

# Catalysts identified
for cat in result.catalysts:
    print(f"[{cat.catalyst.value}] '{cat.matched_phrase}' -> {cat.sentiment_bias.value}")
```

---

## 📚 Dataset & Ingestion Guide

The project includes built-in loaders for the **Financial PhraseBank v.1.0** dataset (Malo et al., 2014) from Aalto University School of Business.

### Understanding the `.txt` and `.csv` Files

| File | Agreement Tier | Sentences | Best Use Case |
| :--- | :---: | :---: | :--- |
| `FinancialPhraseBank/Sentences_AllAgree.txt` | **100%** | 2,264 | **Gold-standard test benchmark**; high-precision trading signals with zero human disagreement. |
| `FinancialPhraseBank/Sentences_75Agree.txt` | **≥ 75%** | 3,453 | **Recommended training balance**; high signal-to-noise ratio with strong sample diversity. |
| `FinancialPhraseBank/Sentences_66Agree.txt` | **≥ 66%** | 4,217 | Intermediate consensus dataset. |
| `FinancialPhraseBank/Sentences_50Agree.txt` | **≥ 50%** | 4,846 | Full majority-vote dataset. |
| `all-data.csv` | **≥ 50%** | 4,846 | CSV-formatted equivalent of 50Agree (`sentiment,sentence`). |

### Loading Datasets Programmatically

```python
from fin_sentiment import PhraseBankLoader

loader = PhraseBankLoader()

# Load 100% agreement gold standard
gold_items = loader.load_agreement_txt(min_agreement=1.00)

# Load CSV dataset
csv_items = loader.load_csv("all-data.csv")

# Create stratified train / val / test splits
train, val, test = loader.create_splits(gold_items, test_size=0.15, val_size=0.15)
```

---

## 🏋️ Model Training & Benchmarks

### Training the Baseline Classifier

Run the automated training script:

```bash
python examples/train_baseline.py
```

#### Benchmark Results (Holdout Test Set on 75% Consensus Data):

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **85.14%** |
| **Balanced Accuracy** | **79.40%** |
| **Macro F1-Score** | **0.8074** |
| **Weighted F1-Score** | **0.8502** |
| **Directional Accuracy (Bull vs Bear)** | **75.51%** |
| **Training Time** | **~1.8 seconds on CPU** |
| **Inference Latency** | **< 1ms per sentence** |

### Fine-Tuning FinBERT Transformer

For maximum contextual understanding, use the transformer training template:

```bash
python examples/train_transformer.py
```

---

## 💻 Command Line Interface (CLI)

```bash
# Analyze a sentence
fin-sentiment analyze "Nvidia revenue surged 122% on data center AI chip demand"

# Output as JSON for integration into pipelines
fin-sentiment analyze "Federal Reserve signals interest rate pause" --json

# Analyze an entire article file
fin-sentiment analyze-article data/sample_article.txt --ticker AAPL

# Train a baseline model on 75% agreement tier
fin-sentiment train --agreement 0.75 --save-path checkpoints/custom_baseline.joblib
```

---

## 🧪 Testing

Run the comprehensive unit test suite:

```bash
pytest -v
```

All 25 unit tests verify:
- Ingestion of `.csv` and `@`-separated `.txt` files across encodings.
- Financial decimal and abbreviation preservation.
- Cashtag, exchange ticker, and corporate entity extraction.
- Catalyst detection and directional scoring boost.
- Baseline model fitting, inference, and serialization.
- Financial metrics computation.
- End-to-end article sentiment analysis.

---

## 🤝 Contributing & Engineering Roadmap

We actively welcome open-source contributions from quantitative researchers, NLP engineers, and algorithmic traders!

- Read our **[Contribution Guidelines](CONTRIBUTING.md)** for developer setup, testing protocols, and PR workflows.
- Explore our **[Architectural Audit & Roadmap (GAPS_AND_ROADMAP.md)](GAPS_AND_ROADMAP.md)** for in-depth documentation of known linguistic edge cases, multi-entity attribution gaps, and open engineering tickets.
- Check our active **[GitHub Issues](https://github.com/ANCLP/finsight/issues)** to pick up a task!

---

## 📄 License & Attribution

- **Library Code**: Released under the [Apache 2.0 License](LICENSE).
- **Financial PhraseBank Dataset**: The raw dataset files in `FinancialPhraseBank/` are authored by Malo et al. (Aalto University School of Business) and licensed under [CC BY-NC-SA 3.0](FinancialPhraseBank/License.txt). If using for academic or research purposes, please cite:
  > Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782-796.
