"""Article ingestion and parsing engine.

Extracts headlines, paragraphs, and sentence streams from financial news feeds,
articles, JSON payloads, or raw strings.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from fin_sentiment.ingestion.text_preprocessor import TextPreprocessor


class ArticleParser:
    """Parses raw financial news articles into clean structured components."""

    @classmethod
    def parse(
        cls,
        source: Union[str, Dict[str, Any]],
        headline: Optional[str] = None,
    ) -> Tuple[Optional[str], str, List[str]]:
        """Parses an article into (headline, cleaned_body, sentences).

        Args:
            source: Raw string article, JSON string, or dict with keys like 'title'/'headline' and 'body'/'content'/'text'.
            headline: Explicit headline if provided separately.

        Returns:
            Tuple of (resolved_headline, cleaned_full_text, list_of_sentences)
        """
        extracted_headline = headline
        body_text = ""

        # Case 1: Dict payload (e.g. from NewsAPI, Bloomberg RSS, or custom scraper)
        if isinstance(source, dict):
            extracted_headline = (
                headline
                or source.get("headline")
                or source.get("title")
                or source.get("name")
            )
            body_text = (
                source.get("body")
                or source.get("content")
                or source.get("text")
                or source.get("description")
                or ""
            )

        # Case 2: String payload (could be raw text or a JSON string)
        elif isinstance(source, str):
            clean_input = source.strip()
            if clean_input.startswith("{") and clean_input.endswith("}"):
                try:
                    parsed_json = json.loads(clean_input)
                    return cls.parse(parsed_json, headline=headline)
                except json.JSONDecodeError:
                    pass

            # Detect if first line looks like a headline (separated by double newline)
            if not extracted_headline and "\n\n" in clean_input:
                parts = clean_input.split("\n\n", 1)
                first_line = parts[0].strip()
                # If first block is short and doesn't end with a period, treat as headline
                if len(first_line) < 150 and not first_line.endswith("."):
                    extracted_headline = first_line
                    body_text = parts[1].strip()
                else:
                    body_text = clean_input
            else:
                body_text = clean_input

        # Clean headline and body
        cleaned_hl = TextPreprocessor.clean_text(extracted_headline) if extracted_headline else None
        cleaned_body = TextPreprocessor.clean_text(body_text)

        # Assemble full text for sentence chunking
        full_text_for_split = f"{cleaned_hl}. {cleaned_body}" if cleaned_hl else cleaned_body
        sentences = TextPreprocessor.split_into_sentences(full_text_for_split)

        return cleaned_hl, cleaned_body, sentences
