"""Integration-style tests for comparison and extraction: real embeddings and generated article data."""

import pytest

from newscompare.claims_util import normalize_claim, normalize_claims
from newscompare.compare import compare_claims
from newscompare.llm_dataset import _parse_story_and_claims_from_response


# --- Claim normalization ---


def test_normalize_claim_trim_and_collapse() -> None:
    assert normalize_claim("  One  \n  two  \n  three  ") == "One two three"
    assert normalize_claim("") == ""
    assert normalize_claim(None) == ""


def test_normalize_claims_dedup_and_cap() -> None:
    out = normalize_claims(["  A  ", "A", "B", "B", "C"])
    assert out == ["A", "B", "C"]
    assert len(normalize_claims(["x"] * 25)) <= 20


# --- Comparison with real embeddings (slower) ---


@pytest.mark.slow
def test_compare_real_embeddings_same_fact_agreed() -> None:
    """With real embeddings, two sources with same fact should yield agreed (threshold 0.74)."""
    article_claims = [
        ("art1", "BBC", "Israel launched strikes on Tehran on April 19."),
        ("art2", "NYT", "Israel launched strikes on Tehran on April 19."),
    ]
    result = compare_claims(article_claims, claim_match_threshold=0.74)
    assert len(result) >= 1
    agreed = [r for r in result if r.label == "agreed"]
    assert len(agreed) >= 1, "Same fact from two sources should produce at least one agreed"
    assert agreed[0].claim_text in (
        "Israel launched strikes on Tehran on April 19.",
        article_claims[0][2],
    )


@pytest.mark.slow
def test_compare_real_embeddings_paraphrase_agreed() -> None:
    """Paraphrased same fact from two sources should often yield agreed at 0.74."""
    article_claims = [
        ("art1", "SourceA", "The minister resigned on Monday."),
        ("art2", "SourceB", "On Monday the minister announced his resignation."),
    ]
    result = compare_claims(article_claims, claim_match_threshold=0.74)
    agreed = [r for r in result if r.label == "agreed"]
    uncorr = [r for r in result if r.label == "uncorroborated"]
    # We expect either one agreed or two uncorroborated depending on embedding similarity
    assert len(result) >= 1
    assert len(agreed) <= 1 and (len(agreed) + len(uncorr)) == len(result)


@pytest.mark.slow
def test_compare_real_embeddings_different_facts_mostly_uncorroborated() -> None:
    """Unrelated facts from different sources should stay uncorroborated."""
    article_claims = [
        ("art1", "A", "The euro rose against the dollar."),
        ("art2", "B", "A fire broke out at the factory."),
        ("art3", "C", "The president signed a new law."),
    ]
    result = compare_claims(article_claims, claim_match_threshold=0.74)
    agreed = [r for r in result if r.label == "agreed"]
    assert len(agreed) == 0, "Unrelated facts should not be grouped as agreed"
    assert len(result) == 3


# --- Generated article / claim scenarios ---


def test_compare_generated_same_claim_two_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated scenario: two articles, same claim text, different sources -> one agreed."""
    def fake_embed(texts: list, model_name: str = ""):
        return __import__("numpy").ones((len(texts), 64), dtype="float32") / 8.0

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    article_claims = [
        ("gen1", "GeneratedSource1", "Event X occurred on date Y."),
        ("gen2", "GeneratedSource2", "Event X occurred on date Y."),
    ]
    result = compare_claims(article_claims, claim_match_threshold=0.74)
    assert len(result) == 1
    assert result[0].label == "agreed"
    assert len(result[0].matched_sources) == 1
    assert result[0].matched_sources[0] in ("GeneratedSource1", "GeneratedSource2")


def test_compare_generated_three_sources_two_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated: three sources, two say same thing, one says different -> one agreed, one uncorroborated."""
    def fake_embed(texts: list, model_name: str = ""):
        # First two claims same vector, third different
        import numpy as np
        n = len(texts)
        out = np.zeros((n, 64), dtype=np.float32)
        out[0] = 1.0 / 8.0
        out[1] = 1.0 / 8.0
        out[2, 1] = 1.0
        out[2] /= np.linalg.norm(out[2])
        return out

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    article_claims = [
        ("a1", "S1", "The deal was signed."),
        ("a2", "S2", "The deal was signed."),
        ("a3", "S3", "Weather was sunny."),
    ]
    result = compare_claims(article_claims, claim_match_threshold=0.72)
    agreed = [r for r in result if r.label == "agreed"]
    uncorr = [r for r in result if r.label == "uncorroborated"]
    assert len(agreed) == 1
    assert len(uncorr) == 1
    assert agreed[0].claim_text == "The deal was signed."
    assert uncorr[0].claim_text == "Weather was sunny."


# --- Extraction parser (story + claims from LLM-like output) ---


def test_parse_story_and_claims_json() -> None:
    raw = '''{"summary": "The minister resigned. The government announced a successor.", "claims": ["The minister resigned on Monday.", "A new minister was appointed."]}'''
    summary, claims, incident = _parse_story_and_claims_from_response(raw)
    assert "minister resigned" in summary
    assert "successor" in summary or "government" in summary
    assert "resigned on Monday" in claims[0]
    assert len(claims) == 2
    assert incident.get("action") == ""


def test_parse_story_and_claims_structured_incident() -> None:
    raw = """{"synopsis":"Leaders met in Brussels.","incident":{"action":"EU leaders agreed on a joint statement.","driver":"Not stated in the text.","outcome":"Statement to be signed Friday.","timeframe":"March 2026.","actor":"European Council","affected":"Member states","additional_context":"None."},"claims":["The vote was unanimous."]}"""
    synopsis, claims, incident = _parse_story_and_claims_from_response(raw)
    assert "Brussels" in synopsis
    assert incident.get("actor") == "European Council"
    assert len(claims) == 1


def test_parse_story_and_claims_claims_only() -> None:
    raw = '''{"claims": ["Fact one here.", "Fact two here."]}'''
    summary, claims, incident = _parse_story_and_claims_from_response(raw)
    assert summary == ""
    assert len(claims) == 2
    assert "Fact one" in claims[0]


def test_parse_story_and_claims_fallback_lines() -> None:
    raw = """
    First factual sentence here with enough length.
    Second factual sentence here as well.
    """
    summary, claims, incident = _parse_story_and_claims_from_response(raw)
    assert summary == ""
    assert len(claims) >= 1
    assert not any(incident.values())
