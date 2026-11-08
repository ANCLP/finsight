"""Tests for the Discourse & Syntactic Clause Deconstructor."""

from fin_sentiment.ingestion.discourse_deconstructor import DiscourseDeconstructor


def test_initial_contrastive_marker():
    text = "Although revenue surged 32% beating Wall Street targets, the firm reported severe litigation liabilities that wiped out cash flow."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.is_contrastive is True
    assert "Although" in res.contrastive_marker or "although" in res.contrastive_marker
    assert "litigation liabilities" in res.operative_clause
    assert "revenue surged" in res.concession_clause
    assert res.operative_weight == 0.80
    assert res.concession_weight == 0.20


def test_mid_contrastive_marker():
    text = "Company reported stellar revenue, but missed consensus estimates significantly."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.is_contrastive is True
    assert "missed consensus" in res.operative_clause
    assert "stellar revenue" in res.concession_clause


def test_directional_metric_debt_expanded():
    text = "The automotive group rapidly expanded borrowings to fund acquisition."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.directional_metric_override is not None
    assert res.directional_metric_override < 0  # Bearish leverage


def test_directional_metric_debt_reduced():
    text = "Company successfully repaid debt ahead of schedule and reduced overall borrowings."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.directional_metric_override is not None
    assert res.directional_metric_override > 0  # Bullish deleveraging


def test_scare_quotes_detection():
    text = "CEO touts 'revolutionary' milestone after third consecutive quarterly delivery miss."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.has_scare_quotes is True


def test_rumor_denial_inversion():
    text = "Management vehemently denied media reports of an accounting fraud probe."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.is_rumor_denial is True


def test_simple_non_contrastive_sentence():
    text = "TCS reported a 14% growth in operating profits for the second quarter."
    res = DiscourseDeconstructor.deconstruct(text)
    assert res.is_contrastive is False
    assert res.operative_clause == text
    assert res.concession_clause is None
    assert res.has_scare_quotes is False
    assert res.is_rumor_denial is False
