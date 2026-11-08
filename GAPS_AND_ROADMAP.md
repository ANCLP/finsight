# 🔬 Architectural Audit: Gaps, Loopholes & Engineering Roadmap

This document provides a comprehensive technical audit of the `fin-sentiment` codebase. It outlines known algorithmic edge cases, semantic loopholes, linguistic failure modes, and performance trade-offs.

Each item includes:
1. **GitHub Issue Reference** (tracked in the repository issue tracker)
2. **Component & Code Location**
3. **Root Cause Analysis**
4. **Reproducible Code Demonstration**
5. **Trading / Quantitative Impact**
6. **Proposed Architectural Remedy**

---

## Table of Contents
- [1. Entity Extraction & Multi-Stock Attribution](#1-entity-extraction--multi-stock-attribution)
  - [Gap 1.1: Standalone Uppercase Ticker Suppression](#gap-11-standalone-uppercase-ticker-suppression-issue-1)
  - [Gap 1.2: Multi-Entity Headline Sentiment Conflation](#gap-12-multi-entity-headline-sentiment-conflation-issue-2)
  - [Gap 1.3: Dollar Amount vs. Cashtag False Positives](#gap-13-dollar-amount-vs-cashtag-false-positives-fixed)
  - [Gap 1.4: Lack of Discourse Coreference & Context Continuity](#gap-14-lack-of-discourse-coreference--context-continuity)
  - [Gap 1.5: Hardcoded Static Company Dictionary](#gap-15-hardcoded-static-company-dictionary-issue-7)
- [2. Discourse Deconstruction & Negation Blindness](#2-discourse-deconstruction--negation-blindness)
  - [Gap 2.1: Precedence Inversion Between Metric Overrides and Contrastive Clauses](#gap-21-precedence-inversion-between-metric-overrides-and-contrastive-clauses-issue-4)
  - [Gap 2.2: Negation Blindness in Directional Metric Pairing](#gap-22-negation-blindness-in-directional-metric-pairing-issue-3)
  - [Gap 2.3: Negation-Blind Catalyst Matching](#gap-23-negation-blind-catalyst-matching-issue-3)
  - [Gap 2.4: Sequential First-Match Short-Circuiting](#gap-24-sequential-first-match-short-circuiting)
- [3. Scoring Dynamics & Probability Recalibration](#3-scoring-dynamics--probability-recalibration)
  - [Gap 3.1: Catalyst Probability Double-Counting in Article Aggregation](#gap-31-catalyst-probability-double-counting-in-article-aggregation-issue-6)
  - [Gap 3.2: Non-Linear Probability Shifting Artifacts](#gap-32-non-linear-probability-shifting-artifacts)
  - [Gap 3.3: Profit vs. Loss Asymmetry in Baseline Bag-of-Words](#gap-33-profit-vs-loss-asymmetry-in-baseline-bag-of-words)
- [4. Runtime Footprint, Cold-Start & Packaging](#4-runtime-footprint-cold-start--packaging)
  - [Gap 4.1: Unconditional Free-Tier LLM Arbiter Invocation](#gap-41-unconditional-free-tier-llm-arbiter-invocation-issue-5)
  - [Gap 4.2: Git-Ignored Checkpoint Cold-Start Latency](#gap-42-git-ignored-checkpoint-cold-start-latency)
  - [Gap 4.3: Missing Packaging Extras for ONNX Runtime](#gap-43-missing-packaging-extras-for-onnx-runtime-issue-8)

---

## 1. Entity Extraction & Multi-Stock Attribution

### Gap 1.1: Standalone Uppercase Ticker Suppression ([Issue #1](https://github.com/ANCLP/finsight/issues/1))
* **Location**: `fin_sentiment/entity/ticker_extractor.py`
* **Root Cause**: The standalone uppercase ticker extraction pass was guarded by `if not detected:`. If even a single cashtag (`$TCS`) or exchange ticker (`NSE:RELIANCE`) was found earlier in the string, bare uppercase tickers (`INFY`, `AAPL`, `MSFT`) were completely skipped.
* **Reproducible Example**:
  ```python
  from fin_sentiment.entity.ticker_extractor import TickerExtractor

  # Prior to fix:
  tickers = TickerExtractor.extract_tickers("$TCS reported record profit while INFY missed revenue.")
  # Result: ['TCS']  <-- INFY was silently omitted!
  ```
* **Trading Impact**: Per-entity ratings completely fail to evaluate companies that lack a dollar sign prefix in multi-stock comparison sentences.
* **Remedy**: Decouple standalone uppercase validation from the cashtag detection state. Cross-reference candidates against `COMMON_COMPANY_TICKERS.values()` and `TICKER_STOPWORDS` independently. *(Patch committed in branch `feat/contributing-and-audit-fixes`)*.

---

### Gap 1.2: Multi-Entity Headline Sentiment Conflation ([Issue #2](https://github.com/ANCLP/finsight/issues/2))
* **Location**: `fin_sentiment/analyzer.py` (`analyze_article` lines 247-268)
* **Root Cause**: When a sentence or headline mentions multiple distinct stock symbols (e.g. *"TCS Beats Expectations While Infosys Faces Margin Squeeze"*), the entire sentence is analyzed once by the classifier as a unified block (`BULLISH (+0.46)` due to `EARNINGS_BEAT`). That identical score is then assigned to both `TCS` and `INFY`.
* **Reproducible Example**:
  ```python
  from fin_sentiment import FinancialNewsSentimentAnalyzer

  analyzer = FinancialNewsSentimentAnalyzer()
  article = """Tech Earnings Divergence: TCS Beats Expectations While Infosys Faces Margin Squeeze

  Tata Consultancy Services ($TCS) reported a 14.8% YoY jump in quarterly net profit.
  On the other hand, Infosys ($INFY) missed revenue expectations.
  """
  result = analyzer.analyze_article(article)
  print(result.entities["INFY"].sentiment.label)
  # Output: BULLISH (+0.09)  <-- INFY is marked BULLISH despite poor results!
  ```
* **Trading Impact**: Cross-talk between tickers causes bullish momentum from a leader to bleed into a lagging competitor, producing dangerous false buy signals.
* **Remedy**: Implement syntactic clause-level entity attribution:
  1. Deconstruct multi-entity sentences along contrastive separators (`while`, `whereas`, `but`, `in contrast`).
  2. Associate each entity specifically with its corresponding clause.

---

### Gap 1.3: Dollar Amount vs. Cashtag False Positives (FIXED)
* **Location**: `fin_sentiment/entity/ticker_extractor.py`
* **Root Cause**: `CASHTAG_RE` previously used `r"\$([A-Za-z0-9\.\_]{1,12})\b"`. Numeric monetary amounts like `$89.5 billion` or `$0.24 dividend` matched as cashtags (`89.5`, `0.24`) and were registered as phantom stock tickers.
* **Remedy**: Enforce that the first character following the dollar sign must be alphabetic: `r"\$([A-Za-z][A-Za-z0-9\.\_]{0,11})\b"`.

---

### Gap 1.4: Lack of Discourse Coreference & Context Continuity
* **Location**: `fin_sentiment/ingestion/article_parser.py` & `analyzer.py`
* **Root Cause**: Financial journalists typically mention the subject company in paragraph sentence 1, and describe operating metrics in subsequent sentences without repeating the company name:
  > *"Infosys ($INFY) reported third-quarter results. Operating margin came down to 20.5% from 22.1%, and management announced plans to trim hiring."*
  Because sentence 2 has no explicit ticker or cashtag, it is dropped from `ticker_sentence_map`.
* **Trading Impact**: Crucial granular financial data (margins, guidance, layoffs) is excluded from entity-level ratings unless the company name is redundantly restated in every single sentence.
* **Remedy**: Introduce a sliding discourse paragraph context: inherit the active entity from preceding sentences within the same paragraph unless an explicit subject change or transition occurs.

---

### Gap 1.5: Hardcoded Static Company Dictionary ([Issue #7](https://github.com/ANCLP/finsight/issues/7))
* **Location**: `fin_sentiment/entity/ticker_extractor.py` (`COMMON_COMPANY_TICKERS`)
* **Root Cause**: The ticker lookup table contains only 45 hardcoded bluechips. There is no mechanism to register custom tickers or ingest exchange ticker masters (e.g. NSE, BSE, NASDAQ, NYSE, LSE) dynamically.
* **Remedy**: Expose public registration methods:
  ```python
  TickerExtractor.register_ticker("ZOMATO", aliases=["zomato", "zomato limited"])
  TickerExtractor.load_from_csv("path/to/nse_symbols.csv")
  ```

---

## 2. Discourse Deconstruction & Negation Blindness

### Gap 2.1: Precedence Inversion Between Metric Overrides and Contrastive Clauses ([Issue #4](https://github.com/ANCLP/finsight/issues/4))
* **Location**: `fin_sentiment/analyzer.py` (lines 137-160)
* **Root Cause**: In `analyze_text()`, `decon.directional_metric_override` is checked and returned *before* checking `decon.is_contrastive`:
  ```python
  if decon.directional_metric_override is not None:
      return SentimentScore(...)  # Returns immediately!

  if decon.is_contrastive and decon.concession_clause:
      # Never reached if concession mentions debt/costs!
  ```
* **Reproducible Example**:
  ```python
  # Sentence starts with concession about debt, but the operative clause is wildly positive:
  text = "Although debt expanded to fund capex, revenues surged 40% beating all analyst targets."
  score = analyzer.analyze_text(text)
  print(score.label, score.bull_bear_score)
  # Output: BEARISH (-0.55)
  ```
* **Trading Impact**: High-growth companies investing aggressively in capex or expanding leverage to fund lucrative acquisitions are misclassified as Bearish distress cases.
* **Remedy**: Evaluate `is_contrastive` first, or restrict directional metric detection strictly to the operative clause.

---

### Gap 2.2: Negation Blindness in Directional Metric Pairing ([Issue #3](https://github.com/ANCLP/finsight/issues/3))
* **Location**: `fin_sentiment/ingestion/discourse_deconstructor.py` (`_check_directional_metrics`)
* **Root Cause**: Matches liability nouns (`debt`, `loss`, `borrowings`, `costs`) within 4 tokens of change verbs (`expand`, `increase`, `reduce`) without verifying whether the verb is preceded by negation tokens (`not`, `did not`, `failed to`, `no`, `neither`).
* **Reproducible Example**:
  ```python
  analyzer.analyze_text("The company did not increase its debt.")
  # Output: BEARISH (-0.55)

  analyzer.analyze_text("Debt failed to expand this quarter.")
  # Output: BEARISH (-0.55)
  ```
* **Trading Impact**: Explicit deleveraging announcements and debt stabilization statements are assigned severe bearish penalties.
* **Remedy**: Inspect intervening tokens between the metric noun and verb for negation markers (`did not`, `didn't`, `failed to`, `prevented from`, `no`).

---

### Gap 2.3: Negation-Blind Catalyst Matching ([Issue #3](https://github.com/ANCLP/finsight/issues/3))
* **Location**: `fin_sentiment/entity/financial_catalysts.py` (`detect_catalysts`)
* **Root Cause**: Catalysts are matched via regular expressions without checking for negative context or denials within the phrase window.
* **Reproducible Example**:
  ```python
  from fin_sentiment.entity.financial_catalysts import CatalystExtractor

  # 1. Negative catalyst triggered when no event occurred:
  cats = CatalystExtractor.detect_catalysts("Management confirmed that no layoffs are planned.")
  # -> Matches LAYOFF_RESTRUCTURING (Bearish -0.14)

  # 2. Positive catalyst triggered when target was missed:
  cats = CatalystExtractor.detect_catalysts("The company did not beat consensus estimates.")
  # -> Matches EARNINGS_BEAT (Bullish +0.18)

  # 3. Regulatory penalty triggered when company was cleared:
  cats = CatalystExtractor.detect_catalysts("Regulators confirmed no fine or penalty will be imposed.")
  # -> Matches REGULATORY_FINE (Bearish -0.18)
  ```
* **Trading Impact**: Catalyst boosts (up to +/-0.30 polarity) apply in reverse when companies successfully avoid penalties or fail to hit benchmarks.
* **Remedy**: Add negative lookbehinds or a pre-match negation filter scanning 3-4 words preceding the catalyst trigger words.

---

### Gap 2.4: Sequential First-Match Short-Circuiting
* **Location**: `fin_sentiment/ingestion/discourse_deconstructor.py` (lines 194-211)
* **Root Cause**: `_check_directional_metrics` iterates through tokens sequentially and returns upon the very first match:
  ```python
  for i, word in enumerate(tokens):
      if word in cls.LIABILITY_METRIC_NOUNS:
          ...
          return -0.55, ... # Returns on first hit
  ```
* **Reproducible Example**:
  In compound disclosures such as *"Raw material costs fell, but freight and logistics expenses jumped"*, whichever noun appears first in the token sequence dictates the polarity (+0.50 Bullish), ignoring the subsequent clause.
* **Remedy**: Aggregate all metric-verb pairs across the sentence and compute a weighted net balance rather than short-circuiting on token index 0.

---

## 3. Scoring Dynamics & Probability Recalibration

### Gap 3.1: Catalyst Probability Double-Counting in Article Aggregation ([Issue #6](https://github.com/ANCLP/finsight/issues/6))
* **Location**: `fin_sentiment/analyzer.py` (lines 280-312)
* **Root Cause**:
  1. `analyze_text(sent)` computes baseline probabilities and shifts them using `catalyst_boost`.
  2. In `analyze_article()`, `overall_sentiment` averages those *already-shifted* probabilities, and then passes them to `SentimentScore.from_probabilities(avg_probs, catalyst_boost=overall_catalyst_boost)`.
  3. `entities[sym]` repeats the exact same pattern: averages already-boosted sentence probabilities and applies `sym_boost` a second time.
* **Trading Impact**: Articles containing multiple catalysts experience extreme probability saturation, skewing confidence levels artificially toward 1.00 or -1.00.
* **Remedy**: Store unboosted base model probabilities in `SentenceAnalysis`, average base probabilities across sentences, and apply the aggregate catalyst boost exactly once at the top level.

---

### Gap 3.2: Non-Linear Probability Shifting Artifacts
* **Location**: `fin_sentiment/types.py` (`from_probabilities` lines 139-147)
* **Root Cause**:
  ```python
  if catalyst_boost > 0:
      shift = min(p_neut * 0.75, catalyst_boost)
      p_bull += shift
      p_neut -= shift
  ```
  The maximum possible boost is clamped to `75%` of the neutral probability (`p_neut`). If the baseline model is already confident (e.g. `p_neut = 0.04`), `shift` cannot exceed `0.03`, rendering a high-conviction catalyst (+0.20) completely ineffective. Conversely, if `p_neut` is large, the boost takes a massive leap.
* **Trading Impact**: Catalyst influence varies unpredictably based on the model's neutral probability entropy rather than catalyst conviction.
* **Remedy**: Shift probability mass proportionally from opposing sentiment (take from `p_bear` when boosting `p_bull`) or calibrate via logit space additions before softmax.

---

### Gap 3.3: Profit vs. Loss Asymmetry in Baseline Bag-of-Words
* **Location**: `fin_sentiment/models/baseline_model.py`
* **Root Cause**: In TF-IDF representations:
  - *"Operating profit rose 10%"* -> Strong Bullish signal.
  - *"Operating loss rose 10%"* -> Often classified as Bullish by bag-of-words classifiers because "rose" and "operating" carry high positive training weights in PhraseBank.
* **Remedy**: Ensure `LIABILITY_METRIC_NOUNS` includes `loss`, `net loss`, and `losses`, and verify that directional metric pairing properly intercepts loss expansion.

---

## 4. Runtime Footprint, Cold-Start & Packaging

### Gap 4.1: Unconditional Free-Tier LLM Arbiter Invocation ([Issue #5](https://github.com/ANCLP/finsight/issues/5))
* **Location**: `fin_sentiment/analyzer.py` (line 112) & `fin_sentiment/models/llm_arbiter.py` (line 91)
* **Root Cause**:
  `analyzer.py` calls:
  ```python
  if self.arbiter.should_arbitrate(text, baseline_conf=0.5):
  ```
  And `llm_arbiter.py` defines:
  ```python
  if 0.40 <= baseline_conf <= 0.62:
      return True
  ```
  Because `baseline_conf=0.5` was passed as a hardcoded argument, the ambiguous confidence condition evaluates to `True` for every single text analyzed.
* **Trading Impact**: If a user sets `GEMINI_API_KEY`, the ultra-fast (<1ms) baseline model is never used. Every sentence initiates a blocking 6-second HTTP call, instantly exceeding Google AI Studio's 15 RPM free tier on standard articles.
* **Remedy**: Run baseline model inference first, and supply the *actual* calculated confidence score to `should_arbitrate()`.

---

### Gap 4.2: Git-Ignored Checkpoint Cold-Start Latency
* **Location**: `.gitignore` & `fin_sentiment/analyzer.py` (`_auto_fit_baseline`)
* **Root Cause**: `.gitignore` contains `checkpoints/` and `*.joblib`. A freshly cloned repository lacks `baseline_phrasebank.joblib`. When `FinancialNewsSentimentAnalyzer()` initializes, it triggers `_auto_fit_baseline()` on `all-data.csv` (taking 2-3 seconds) but never serializes the fitted pipeline.
* **Remedy**:
  1. Whitelist the lightweight default model in `.gitignore`: `!checkpoints/baseline_phrasebank.joblib`.
  2. Or have `_auto_fit_baseline()` persist the fitted artifact to `checkpoints/` on first run.

---

### Gap 4.3: Missing Packaging Extras for ONNX Runtime ([Issue #8](https://github.com/ANCLP/finsight/issues/8))
* **Location**: `pyproject.toml`
* **Root Cause**: `fin_sentiment/models/onnx_model.py` requires `onnxruntime`, but it was omitted from `pyproject.toml`.
* **Remedy**: Add `[project.optional-dependencies]` entry `onnx = ["onnxruntime>=1.15.0"]` and include in `all`. *(Patch committed in branch `feat/contributing-and-audit-fixes`)*.
