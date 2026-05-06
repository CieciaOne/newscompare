"""Topic headline fallback and LLM label normalization."""

import numpy as np
import pytest

from newscompare.topic_extraction import (
    _fallback_label_from_titles,
    _merge_similar_topic_labels,
    _normalize_label_llm_response,
)


def test_fallback_label_from_titles_picks_longest() -> None:
    out = _fallback_label_from_titles(["", "Short", "Los Angeles wildfire forces evacuations downtown"])
    assert "wildfire" in out.lower() or "Los" in out


def test_normalize_label_json_object() -> None:
    assert "Donbas" in _normalize_label_llm_response('{"label":"War in Donbas Update","x":1}')


def test_normalize_label_plain_line() -> None:
    assert _normalize_label_llm_response("  Central Bank Holds Rates  \nextra") == "Central Bank Holds Rates"


def test_merge_similar_topic_labels_unions(monkeypatch: pytest.MonkeyPatch) -> None:
    import newscompare.topic_extraction as te

    topics = [
        ("tid-a", "Russia Ukraine Diplomatic Talks", "slug-a"),
        ("tid-b", "Ukraine Russia Diplomacy Update", "slug-b"),
    ]
    assign = {"x": ["tid-a"], "y": ["tid-b"]}

    def fake_embed(texts: list, model_name: str = ""):
        return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(te, "embed_texts", fake_embed)
    new_t, new_a = _merge_similar_topic_labels(topics, assign, "dummy", threshold=0.5)
    assert len(new_t) == 1
    assert new_a["x"] == new_a["y"]
    assert len(new_a["x"]) == 1
