"""Serverless Free-Tier Semantic Arbiter using Gemini 1.5 Flash.

Provides human-level financial reasoning for:
1. Multi-entity sentiment cross-talk (Aspect-Based Sentiment Analysis)
2. Expectations vs. Actuals ("Beats" vs "Misses")
3. Implicit regulatory distress catalysts without explicit sentiment words

Operates strictly within Google AI Studio's 100% Free Tier (1,500 requests/day, $0.00/mo)
with 24-hour deduplication caching and seamless offline fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from fin_sentiment.types import SentimentLabel, SentimentScore

logger = logging.getLogger(__name__)


class GeminiFinancialArbiter:
    """Zero-cost LLM semantic arbiter for complex financial headlines."""

    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"

    # In-memory TTL cache for LLM arbitration verdicts
    _cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    _cache_lock = Lock()

    @classmethod
    def _get_cached(cls, key: str) -> Optional[Dict[str, Any]]:
        with cls._cache_lock:
            entry = cls._cache.get(key)
            if entry:
                expires_at, data = entry
                if time.time() < expires_at:
                    return data
                cls._cache.pop(key, None)
            return None

    @classmethod
    def _set_cached(cls, key: str, data: Dict[str, Any], ttl_seconds: int = 86400) -> None:
        with cls._cache_lock:
            cls._cache[key] = (time.time() + ttl_seconds, data)

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    def is_available(self) -> bool:
        """Returns True if a valid Google AI API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def should_arbitrate(self, headline: str, extracted_entities: Optional[list] = None, baseline_conf: float = 0.5) -> bool:
        """Determines if a headline contains architectural red flags requiring LLM arbitration."""
        lower = headline.lower()

        # 1. Multi-entity cross-talk (2+ distinct tickers mentioned)
        if extracted_entities and len(set(extracted_entities)) >= 2:
            return True

        # 2. Consensus expectation comparison terms ("beat vs miss")
        expectation_markers = [
            "missed consensus", "misses estimates", "beat consensus", "topped estimates",
            "fell short of estimates", "lagged forecasts", "guidance lowered", "guidance raised",
            "analyst expectations", "street views", "consensus view",
        ]
        if any(marker in lower for marker in expectation_markers):
            return True

        # 3. Implicit distress catalysts with ambiguous wording
        distress_markers = [
            "auditor resigns", "forensic audit", "show-cause", "subpoena", "default on coupon",
            "delay in filing", "misses deadline", "accounting irregularities",
        ]
        if any(marker in lower for marker in distress_markers):
            return True

        # 4. Low baseline confidence (ambiguous zone)
        if 0.40 <= baseline_conf <= 0.62:
            return True

        return False

    def analyze(self, headline: str, target_ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Evaluates a complex headline via Gemini 1.5 Flash."""
        if not self.is_available():
            return None

        # Check 24-hour cache
        cache_key = f"gemini_arbiter_{headline.strip().lower()}_{target_ticker or 'ALL'}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        prompt = f"""You are a senior quantitative financial analyst. Analyze this market headline with high financial precision:

Headline: "{headline}"
Target Company Ticker (if applicable): {target_ticker or 'None specified'}

Analyze:
1. Operative sentiment (bullish, bearish, or neutral).
2. Continuous score (-1.0 to +1.0) where -1.0 is severe crisis/crash, +1.0 is historic breakout.
3. If expectations are mentioned, prioritize actual performance vs consensus (e.g., revenue growing 20% but missing expectations by 5% is BEARISH).
4. If multiple companies are mentioned, isolate sentiment specifically for the Target Company.
5. If regulatory distress (auditor resigns, forensic probe, debt default) is mentioned, score as strongly bearish.

Output ONLY a JSON object in this exact format:
{{
  "label": "bullish" | "bearish" | "neutral",
  "bull_bear_score": float,
  "confidence": float,
  "reasoning": "one clear sentence explanation",
  "primary_catalyst": "string",
  "entity_sentiments": {{"TICKER": float}}
}}
"""

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        url = self.API_URL_TEMPLATE.format(key=self.api_key)
        headers = {"Content-Type": "application/json"}

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            candidates = data.get("candidates", [])
            if not candidates:
                return None

            content_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            parsed = json.loads(content_text)

            # Validate and normalize
            result = {
                "label": parsed.get("label", "neutral").lower(),
                "bull_bear_score": float(parsed.get("bull_bear_score", 0.0)),
                "confidence": float(parsed.get("confidence", 0.85)),
                "reasoning": str(parsed.get("reasoning", "")),
                "primary_catalyst": str(parsed.get("primary_catalyst", "MARKET_INTELLIGENCE")),
                "entity_sentiments": parsed.get("entity_sentiments", {}),
                "source": "gemini_1.5_flash",
            }

            # Cache successful response
            self._set_cached(cache_key, result, ttl_seconds=86400)
            return result

        except Exception as exc:
            logger.warning("Gemini financial arbitration call failed: %s. Falling back to local engine.", exc)
            return None
