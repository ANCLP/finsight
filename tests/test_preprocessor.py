"""Tests for financial text cleaning and sentence segmentation."""

from __future__ import annotations

from fin_sentiment.ingestion.article_parser import ArticleParser
from fin_sentiment.ingestion.text_preprocessor import TextPreprocessor


def test_clean_text():
    raw = "Operating margin came down to  2.4 %  from 5.7 %  .\n\n"
    cleaned = TextPreprocessor.clean_text(raw)
    assert cleaned == "Operating margin came down to 2.4% from 5.7% ."


def test_split_into_sentences_preserves_decimals():
    text = (
        "Operating profit rose to EUR 14.5 mn from EUR 8.2 mn. "
        "The operating margin improved to 7.8% from 5.4% in 2022. "
        "The board recommended a dividend."
    )
    sentences = TextPreprocessor.split_into_sentences(text)
    assert len(sentences) == 3
    assert "EUR 14.5 mn" in sentences[0]
    assert "7.8%" in sentences[1]


def test_split_into_sentences_preserves_company_suffixes():
    text = (
        "Apple Inc. reported quarterly revenue of $90B today. "
        "Meanwhile, Elcoteq SE announced a strategic reorganization."
    )
    sentences = TextPreprocessor.split_into_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Apple Inc.")
    assert sentences[1].startswith("Meanwhile, Elcoteq SE")


def test_article_parser():
    article_text = (
        "Tata Motors Q3 Results Surge on JLR Turnaround\n\n"
        "Tata Motors Ltd. reported a 137% jump in quarterly consolidated net profit to Rs 7,025 crore. "
        "Revenue from operations rose 25% YoY to Rs 1.10 lakh crore. "
        "Commercial vehicle margins expanded by 180 bps."
    )
    headline, body, sentences = ArticleParser.parse(article_text)
    assert headline == "Tata Motors Q3 Results Surge on JLR Turnaround"
    assert "Rs 7,025 crore" in body
    assert len(sentences) >= 3
