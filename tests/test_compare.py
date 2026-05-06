"""Tests for claim comparison logic."""

import numpy as np
import pytest

from newscompare.compare import _match_claims, compare_claims
from newscompare.compare_pair_heuristic import (
    heuristic_pair_verdict_text,
    likely_same_fact_gate,
    numeric_zero_conflict,
)


def test_match_claims_empty() -> None:
    assert _match_claims([], np.zeros((0, 0)), 0.85) == []


def test_match_claims_single() -> None:
    # one claim, no others
    ac = [("a1", "S1", "The economy grew.")]
    embs = np.array([[1.0, 0.0, 0.0]])
    m = _match_claims(ac, embs, 0.85)
    assert len(m) == 1
    assert m[0] == []


def test_match_claims_two_identical_vectors_match() -> None:
    # Same embedding -> similarity 1.0 -> match at any threshold
    ac = [("a1", "S1", "Israel struck Tehran."), ("a2", "S2", "Israel struck Tehran.")]
    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embs = np.vstack([v, v])
    m = _match_claims(ac, embs, 0.72)
    assert len(m) == 2
    assert 1 in m[0]
    assert 0 in m[1]


def test_match_claims_two_similar_above_threshold() -> None:
    # Cosine sim between unit vectors (1,0,0) and (0.9, 0.44, 0) ~ 0.9
    ac = [("a1", "S1", "X"), ("a2", "S2", "Y")]
    embs = np.array([[1.0, 0.0, 0.0], [0.9, 0.44, 0.0]], dtype=np.float32)
    embs[1] /= np.linalg.norm(embs[1])
    m = _match_claims(ac, embs, 0.72)
    assert 1 in m[0]
    assert 0 in m[1]


def test_match_claims_below_threshold_no_match() -> None:
    # Orthogonal vectors -> sim 0
    ac = [("a1", "S1", "X"), ("a2", "S2", "Y")]
    embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    m = _match_claims(ac, embs, 0.85)
    assert m[0] == []
    assert m[1] == []


def test_detect_conflict_negation() -> None:
    assert heuristic_pair_verdict_text("Sales increased.", "Sales did not increase.", 0.85) == "conflict"
    assert heuristic_pair_verdict_text("Sales increased.", "Sales increased again.", 0.85) == "agree"


def test_detect_conflict_increase_decrease() -> None:
    assert heuristic_pair_verdict_text("Unemployment rose.", "Unemployment fell.", 0.8) == "conflict"
    assert heuristic_pair_verdict_text("Unemployment rose.", "Unemployment rose again.", 0.8) == "agree"


def test_compare_claims_empty() -> None:
    assert compare_claims([]) == []


def test_compare_claims_uncorroborated_single(monkeypatch: pytest.MonkeyPatch) -> None:
    # One article, one claim -> uncorroborated. Mock embeddings to avoid loading model.
    def fake_embed(texts: list, model_name: str = ""):
        return np.random.randn(len(texts), 384).astype(np.float32) * 0.01  # small random

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [("a1", "S1", "The minister resigned.")]
    result = compare_claims(ac, claim_match_threshold=0.99)
    assert len(result) == 1
    assert result[0].label == "uncorroborated"


def test_compare_claims_two_same_claim_agreed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two articles, same claim text -> one agreed cluster (deduplicated)
    def fake_embed(texts: list, model_name: str = ""):
        return np.ones((len(texts), 64), dtype=np.float32) / 8.0

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "BBC", "Israel launched strikes on Tehran."),
        ("a2", "NYT", "Israel launched strikes on Tehran."),
    ]
    result = compare_claims(ac, claim_match_threshold=0.72, claim_pair_heuristic_fallback=True)
    assert len(result) == 1
    assert result[0].label == "agreed"
    assert set(result[0].matched_sources) == {"NYT"} or set(result[0].matched_sources) == {"BBC"}
    assert len(result[0].matched_claims) >= 1


def test_match_claims_same_story_pairs_filters() -> None:
    # Four claims: a1-s1, a1-s2, a2-s1, a2-s2. Same embedding for all -> all match at 0.72.
    # same_story_pairs only (a1,a2) and (a1,a1),(a2,a2). So a1 can match a2 only.
    ac = [
        ("a1", "S1", "X"),
        ("a1", "S1", "Y"),
        ("a2", "S2", "X"),
        ("a2", "S2", "Y"),
    ]
    v = np.ones((1, 4), dtype=np.float32) / 2.0
    embs = np.vstack([v, v, v, v])
    same_story = {("a1", "a2"), ("a2", "a1"), ("a1", "a1"), ("a2", "a2")}
    m = _match_claims(ac, embs, 0.72, same_story_pairs=same_story)
    # Each claim matches the other claim from the other article (different source)
    assert len(m) == 4
    # Indices 0,1 are a1; 2,3 are a2. other_source_matches would filter to different source only.
    # So 0 can match 2 or 3 (a2), 1 can match 2 or 3, etc.
    assert 2 in m[0] or 3 in m[0]
    assert 0 in m[2] or 1 in m[2]


def test_match_claims_same_story_pairs_blocks_when_different_story() -> None:
    # a1 and a2 not in same_story -> no cross-match even with identical embeddings
    ac = [
        ("a1", "S1", "Same text."),
        ("a2", "S2", "Same text."),
    ]
    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    embs = np.vstack([v, v])
    same_story = {("a1", "a1"), ("a2", "a2")}  # only same-article pairs
    m = _match_claims(ac, embs, 0.72, same_story_pairs=same_story)
    assert m[0] == []
    assert m[1] == []


def test_likely_same_fact_death_vs_links() -> None:
    # Different events: death vs relations -> not same fact
    assert likely_same_fact_gate(
        "Ali Chamenei zginął.",
        "Modżtaba Chamenei ma bliskie powiązania z IRGC.",
    ) is False
    assert likely_same_fact_gate(
        "Modżtaba Chamenei ma powiązania z IRGC.",
        "Najwyższy przywódca Ali Chamenei zginął.",
    ) is False


def test_likely_same_fact_both_death() -> None:
    assert likely_same_fact_gate("Leader died.", "The president was killed.") is True


def test_compare_claims_with_summaries_same_story_agreed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two articles with similar story summaries and same fact -> agreed
    def fake_embed(texts: list, model_name: str = ""):
        return np.ones((len(texts), 64), dtype=np.float32) / 8.0

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "BBC", "Strike hit the base."),
        ("a2", "NYT", "The base was hit by a strike."),
    ]
    summaries = {
        "a1": "Iran reported an attack on a military base.",
        "a2": "A military base in Iran was attacked.",
    }
    result = compare_claims(
        ac,
        claim_match_threshold=0.72,
        article_summaries=summaries,
        story_similarity_threshold=0.35,
        claim_pair_heuristic_fallback=True,
    )
    assert len(result) == 1
    assert result[0].label == "agreed"


def test_compare_claims_two_similar_above_threshold_agreed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two claims with high similarity (0.8) and threshold 0.72 -> one agreed cluster
    def fake_embed(texts: list, model_name: str = ""):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        out[0, 0] = 1.0
        if len(texts) > 1:
            out[1, 0] = 0.8
            out[1, 1] = 0.6
            out[1] /= np.linalg.norm(out[1])
        return out

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "S1", "Iran was attacked."),
        ("a2", "S2", "Iran came under attack."),
    ]
    result = compare_claims(ac, claim_match_threshold=0.72, claim_pair_heuristic_fallback=True)
    assert len(result) == 1
    assert result[0].label == "agreed"


def test_compare_claims_similar_but_threshold_too_high_uncorroborated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same 0.8 similarity but threshold 0.9 -> no match -> uncorroborated
    def fake_embed(texts: list, model_name: str = ""):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        out[0, 0] = 1.0
        if len(texts) > 1:
            out[1, 0] = 0.8
            out[1, 1] = 0.6
            out[1] /= np.linalg.norm(out[1])
        return out

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "S1", "Iran was attacked."),
        ("a2", "S2", "Iran came under attack."),
    ]
    result = compare_claims(ac, claim_match_threshold=0.9)
    assert len(result) == 2
    assert result[0].label == "uncorroborated"
    assert result[1].label == "uncorroborated"


def test_compare_claims_negation_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same topic (high sim) but negation -> conflict
    def fake_embed(texts: list, model_name: str = ""):
        # Same vector so they match
        return np.ones((len(texts), 64), dtype=np.float32) / 8.0

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "S1", "Sales increased."),
        ("a2", "S2", "Sales did not increase."),
    ]
    result = compare_claims(ac, claim_match_threshold=0.72, claim_pair_heuristic_fallback=True)
    assert len(result) == 2
    assert result[0].label == "conflict"
    assert result[1].label == "conflict"
    assert result[0].conflicting_claim is not None


def test_compare_claims_same_source_not_agreed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two claims from same source (duplicate or two articles same outlet) -> uncorroborated
    def fake_embed(texts: list, model_name: str = ""):
        return np.ones((len(texts), 64), dtype=np.float32) / 8.0

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "NYT", "The U.S. Embassy in Baghdad was targeted."),
        ("a2", "NYT", "The U.S. Embassy in Baghdad was targeted."),
    ]
    result = compare_claims(ac, claim_match_threshold=0.60, claim_pair_heuristic_fallback=True)
    assert len(result) == 2
    assert result[0].label == "uncorroborated"
    assert result[1].label == "uncorroborated"
    assert result[0].matched_sources == []
    assert result[1].matched_sources == []


def test_compare_claims_matched_sources_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list, model_name: str = ""):
        return np.ones((len(texts), 64), dtype=np.float32) / 8.0

    import newscompare.compare as compare_mod
    monkeypatch.setattr(compare_mod, "embed_texts", fake_embed)

    ac = [
        ("a1", "BBC", "Israel struck oil facilities in Iran."),
        ("a2", "NYT", "Israel struck oil facilities in Iran."),
    ]
    result = compare_claims(ac, claim_match_threshold=0.72, claim_pair_heuristic_fallback=True)
    assert len(result) == 1
    assert result[0].label == "agreed"
    assert set(result[0].matched_sources) == {"NYT"} or set(result[0].matched_sources) == {"BBC"}


def test_detect_numeric_conflict() -> None:
    assert numeric_zero_conflict("11 dead in the strike", "No casualties reported") is True
    assert numeric_zero_conflict("No casualties", "Dozens of casualties") is True
    assert numeric_zero_conflict("No dead", "Multiple dead") is True
    assert numeric_zero_conflict("Both said no casualties", "No casualties") is False
