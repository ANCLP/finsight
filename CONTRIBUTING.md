# 🤝 Contributing to fin-sentiment

Thank you for your interest in contributing to **fin-sentiment**!

`fin-sentiment` is an open-source, production-grade financial news sentiment analysis engine, continuous Bull/Bear polarity indexer, and stock entity attribution framework.

---

## 🎯 Project Mission & Design Philosophy

The primary objective of `fin-sentiment` is to deliver **institutional-grade financial sentiment intelligence while maintaining an ultra-lightweight, high-throughput footprint**:

1. **⚡ Extreme Speed (< 1ms per sample)**: Default models must run locally on standard CPUs with near-zero latency, enabling real-time scanning of live news wires and order-routing feeds.
2. **🪶 Zero Heavy Runtime Dependency by Default**: The core library operates on `scikit-learn`, `numpy`, and `pydantic`. Deep transformers (`torch`, `transformers`) and quantized accelerators (`onnxruntime`) are strictly optional extras.
3. **📊 Continuous Bull/Bear Polarity Indexing (-1.00 to +1.00)**: Binary positive/negative classification is insufficient for quantitative finance. We output continuous calibrated scores paired with market-moving catalysts (earnings beats, margin expansions, defaults).
4. **🏢 Multi-Entity Ticker Attribution**: Financial journalism often compares winners and losers in a single headline (*"TCS profit rose while Infosys lagged"*). Our pipeline separates entity-specific sentiment.

---

## 📜 Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors.

- **Be respectful and constructive**: Treat all maintainers and fellow contributors with professional courtesy.
- **Focus on technical rigor**: Financial NLP is nuanced. We welcome evidence-based critique, benchmarks, and real-world failure cases.
- **Maintain zero financial bias**: All algorithms, catalysts, and scoring weights must be objective, deterministic, and free of promotional tilt toward any specific asset or stock.

---

## 💻 Getting Started & Local Environment Setup

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12**
- **Git**

### 2. Fork and Clone
```bash
# Clone your fork (or repository)
git clone https://github.com/ANCLP/finsight.git
cd finsight
```

### 3. Create a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies in Editable Mode

Install core dependencies along with development and testing tools:
```bash
# Core + Development Tools (pytest, black, ruff, mypy)
pip install -e ".[dev]"
```

#### Optional Component Extras:
```bash
# For PyTorch / FinBERT Transformer fine-tuning
pip install -e ".[torch]"

# For Quantized CPU acceleration via ONNX Runtime
pip install -e ".[onnx]"

# Install everything
pip install -e ".[all]"
```

### 5. Verify the Test Suite
Ensure all unit tests pass before making any changes:
```bash
pytest -v
```

---

## 📦 Architecture & Codebase Map

Familiarize yourself with the package structure:

```
fin_sentiment/
├── types.py                   # Domain models: SentimentScore, EntitySentiment, ArticleAnalysis, CatalystMatch
├── analyzer.py                # Main facade: FinancialNewsSentimentAnalyzer
├── cli.py                     # Command-line interface ('fin-sentiment' / 'finsight')
│
├── ingestion/                 # Data loading and linguistic preprocessing
│   ├── phrasebank_loader.py   # Loader for all-data.csv and Sentences_*Agree.txt
│   ├── text_preprocessor.py   # Preserves decimals (2.4%), suffixes (Ltd., SE), currencies
│   ├── article_parser.py      # Parses articles, extracts headlines & paragraph streams
│   └── discourse_deconstructor.py # Subordinate clauses, scare quotes, and directional metrics
│
├── entity/                    # Entity extraction and catalyst detection
│   ├── ticker_extractor.py    # Cashtags ($AAPL), exchange prefixes (NSE:TCS), company dictionary
│   └── financial_catalysts.py # 16+ market catalysts (earnings beats, margin squeeze, buybacks)
│
├── models/                    # Model backends
│   ├── base.py                # Abstract BaseSentimentModel protocol
│   ├── baseline_model.py      # TF-IDF + Calibrated Logistic Regression (Production default)
│   ├── finbert_model.py       # HuggingFace FinBERT PyTorch adapter (Optional extra)
│   ├── onnx_model.py          # Quantized ONNX INT8 engine (Optional extra)
│   └── llm_arbiter.py         # Google AI Studio Free-Tier Gemini arbiter
│
└── training/                  # Model training and benchmark pipelines
    ├── baseline_trainer.py    # Stratified training pipeline for TF-IDF classifier
    ├── transformer_trainer.py # Fine-tuning template for transformer models
    └── metrics.py             # Accuracy, Balanced Accuracy, Macro F1, and Directional Bull/Bear Accuracy
```

---

## 🛠️ Contribution Workflows

We welcome contributions across several core areas. Below are step-by-step guides for common tasks:

### Area 1: Adding or Tuning Financial Catalysts
Catalysts identify market-moving corporate events and apply a directional conviction boost (-0.30 to +0.30) to the baseline sentiment.

1. Open [`fin_sentiment/entity/financial_catalysts.py`](fin_sentiment/entity/financial_catalysts.py).
2. If adding a new catalyst category, add an entry to `CatalystType` in [`fin_sentiment/types.py`](fin_sentiment/types.py).
3. Add the regular expression pattern, sentiment bias (`SentimentLabel.BULLISH` or `SentimentLabel.BEARISH`), and impact weight (0.10 to 0.20) in `CATALYST_DEFINITIONS`:
   ```python
   (
       CatalystType.CONTRACT_WIN,
       r"\b(?:bags?|wins?|secured|awarded|clinched)\s+(?:order|contract|deal|mandate|project)\b|"
       r"\bmulti-million\s+(?:dollar|euro|rupee)\s+(?:deal|contract)\b",
       SentimentLabel.BULLISH,
       0.14,
   ),
   ```
4. **Important**: Ensure your pattern does not match negated phrases (e.g., *"no contract was awarded"*). See [GAPS_AND_ROADMAP.md](GAPS_AND_ROADMAP.md#gap-23-negation-blind-catalyst-matching-issue-3).
5. Add test coverage in [`tests/test_entity_extractor.py`](tests/test_entity_extractor.py).

---

### Area 2: Expanding Ticker & Entity Recognition
Stock entity attribution maps sentiments to specific tickers.

1. Open [`fin_sentiment/entity/ticker_extractor.py`](fin_sentiment/entity/ticker_extractor.py).
2. Add company name variations and canonical symbols to `COMMON_COMPANY_TICKERS`:
   ```python
   "zomato": "ZOMATO",
   "tata steel": "TATASTEEL",
   "crowdstrike": "CRWD",
   ```
3. If an uppercase word coincides with financial jargon (e.g., `CEO`, `IPO`, `EBITDA`, `BUY`), ensure it is added to `TICKER_STOPWORDS` to prevent false positive entity extraction.
4. Add verification tests in [`tests/test_entity_extractor.py`](tests/test_entity_extractor.py).

---

### Area 3: Improving Discourse & Syntactic Parsing
Financial journalists heavily use contrastive clauses (*"Although Q3 revenue surged, unexpected litigation wiped out cash flow"*).

1. Review [`fin_sentiment/ingestion/discourse_deconstructor.py`](fin_sentiment/ingestion/discourse_deconstructor.py).
2. Add new contrastive markers or directional metric pairs:
   - `LIABILITY_METRIC_NOUNS`: terms where increase = Bearish (e.g., `debt`, `capex`, `npa`, `expenses`).
   - `INCREASE_VERBS` / `DECREASE_VERBS`.
3. Add edge case tests in [`tests/test_discourse.py`](tests/test_discourse.py).

---

### Area 4: Model Training & Benchmarking
To train or re-evaluate models on the Financial PhraseBank consensus tiers:

```bash
# Train baseline on 75% annotator consensus tier
python examples/train_baseline.py

# Or via CLI:
fin-sentiment train --agreement 0.75 --save-path checkpoints/custom_baseline.joblib
```

All trained models should be evaluated using `FinancialMetrics.evaluate(y_true, y_pred)` to track:
- **Macro F1-Score** (target: ≥ 0.80)
- **Balanced Accuracy** (target: ≥ 0.79)
- **Directional Bull/Bear Accuracy** (target: ≥ 75%)

---

## 🗺️ Known Gaps & High-Priority Roadmap

Before starting a significant feature, check our documented gaps and open GitHub issues:

* **[GAPS_AND_ROADMAP.md](GAPS_AND_ROADMAP.md)** contains the complete architectural audit of known edge cases and failure modes.
* **Open Issue Tracker**:
  - [Issue #1](https://github.com/ANCLP/finsight/issues/1): Standalone uppercase ticker extraction suppressed when cashtag is present *(Resolved)*
  - [Issue #2](https://github.com/ANCLP/finsight/issues/2): Multi-entity headline conflation causes incorrect sentiment attribution
  - [Issue #3](https://github.com/ANCLP/finsight/issues/3): Negation blindness in financial catalysts and directional metric pairing
  - [Issue #4](https://github.com/ANCLP/finsight/issues/4): Precedence inversion between metric overrides and contrastive clauses
  - [Issue #5](https://github.com/ANCLP/finsight/issues/5): Hardcoded baseline_conf causes unintended LLM Arbiter calls
  - [Issue #6](https://github.com/ANCLP/finsight/issues/6): Catalyst probability double-counting in article and entity aggregations
  - [Issue #7](https://github.com/ANCLP/finsight/issues/7): Dynamic ticker registration and exchange master list support
  - [Issue #8](https://github.com/ANCLP/finsight/issues/8): Missing onnxruntime dependency and broken Dockerfile *(Resolved)*

If you wish to tackle one of these issues, comment on the issue ticket to let maintainers know you are working on it!

---

## 🔍 Quality Standards & Verification

All contributions must adhere to the following quality checks:

### 1. Code Formatting
Format code using [Black](https://github.com/psf/black):
```bash
black fin_sentiment tests examples
```

### 2. Linting
Lint code using [Ruff](https://github.com/astral-sh/ruff):
```bash
ruff check fin_sentiment tests examples
```

### 3. Type Checking
Verify static typing with [Mypy](https://mypy.readthedocs.io/):
```bash
mypy fin_sentiment
```

### 4. Unit Testing
Every bug fix or new feature **must include at least one corresponding unit test**:
```bash
pytest -v
```

---

## 🔀 Git & Pull Request Guidelines

> [!IMPORTANT]
> **Branch Protection Rule**: The `main` branch is protected. Direct pushes to `main` are disabled. All changes must be submitted via Pull Requests from a feature or fix branch.

### 1. Branch Naming Conventions
Use descriptive, hyphenated branch names prefixed with the change type:
- `feat/add-deleveraging-catalyst`
- `fix/negation-lookbehind`
- `perf/fast-tokenizer-vectorization`
- `docs/clarify-cli-options`
- `test/add-scare-quotes-regression`

### 2. Commit Message Standards
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(entity): add support for BSE 6-digit scrip codes
fix(catalysts): prevent false positive EARNINGS_BEAT on negated statements
test(discourse): add regression test for concession clause precedence
docs: update quickstart guide with multi-ticker examples
```

### 3. Pull Request Submission Checklist
Before opening your Pull Request:
- [ ] Created a dedicated branch from latest `main`.
- [ ] Added or updated unit tests covering all modified paths.
- [ ] All 43+ unit tests pass cleanly (`pytest -v`).
- [ ] Code formatted with `black` and passes `ruff check`.
- [ ] Updated docstrings and relevant markdown documentation.
- [ ] Referenced the relevant GitHub issue (e.g. `Closes #3`).

Thank you for helping make `fin-sentiment` the fastest, most reliable open-source financial sentiment engine!
