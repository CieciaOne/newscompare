"""Tests for structured incident schema helpers."""

from newscompare.story_schema import (
    build_claim_embedding_text,
    build_same_story_embedding_text,
    empty_incident,
    normalize_incident,
    parse_story_incident_json,
)


def test_normalize_incident_aliases() -> None:
    raw = {"Action": "Strike occurred", "When": "Monday", "to_whom": "Civilians"}
    out = normalize_incident(raw)
    assert out["action"] == "Strike occurred"
    assert out["timeframe"] == "Monday"
    assert out["affected"] == "Civilians"


def test_build_claim_embedding_includes_situation() -> None:
    inc = normalize_incident({"action": "Parliament voted.", "outcome": "Bill passed."})
    t = build_claim_embedding_text("Turnout was 90%.", inc)
    assert "Parliament voted" in t
    assert "Atomic claim" in t
    assert "Turnout was 90%" in t


def test_parse_story_incident_json_empty() -> None:
    assert parse_story_incident_json("") == empty_incident()
    assert parse_story_incident_json(None) == empty_incident()


def test_build_same_story_embedding_text() -> None:
    inc = normalize_incident({"actor": "EU", "action": "Summit held"})
    t = build_same_story_embedding_text("Leaders met.", inc)
    assert "Synopsis" in t
    assert "Structured incident" in t
    assert "EU" in t
