"""Topic cluster verdict JSON parsing."""

from newscompare.topic_verdict import TopicClusterVerdict, final_topic_label, parse_topic_verdict_json


def test_parse_clean_json() -> None:
    raw = '{"verdict":"unrelated","label":"","why":"Different subjects"}'
    v = parse_topic_verdict_json(raw)
    assert v.verdict == "unrelated"
    assert v.label == ""
    assert "Different" in v.why


def test_parse_markdown_wrapped() -> None:
    raw = 'Here is JSON:\n{"verdict":"disagree","label":"Casualty Figures Clash","why":"One says dozens dead"}\n'
    v = parse_topic_verdict_json(raw)
    assert v.verdict == "disagree"
    assert "Casualty" in v.label


def test_parse_fenced_json() -> None:
    raw = '```json\n{"verdict":"agree","label":"EU Energy Council","why":"Same meeting"}\n```'
    v = parse_topic_verdict_json(raw)
    assert v.verdict == "agree"
    assert "EU" in v.label or "Energy" in v.label


def test_parse_alternate_topic_key() -> None:
    raw = '{"verdict":"agree","topic":"Gaza Border Incidents","why":""}'
    v = parse_topic_verdict_json(raw)
    assert "Gaza" in v.label


def test_parse_malformed_defaults_agree() -> None:
    v = parse_topic_verdict_json("not json")
    assert v.verdict == "agree"


def test_final_label_disputed_prefix() -> None:
    tv = TopicClusterVerdict("disagree", "Border Incident Claims", "x")
    assert final_topic_label(tv, "Fallback").startswith("Disputed:")


def test_final_label_uses_fallback() -> None:
    tv = TopicClusterVerdict("agree", "", "")
    assert final_topic_label(tv, "My Fallback") == "My Fallback"
