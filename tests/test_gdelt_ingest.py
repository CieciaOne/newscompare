"""Tests for GDELT ingestion helpers (no live HTTP)."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from newscompare.feed_fetcher import FeedEntry
from newscompare.gdelt_ingest import (
    clip_range_to_gdelt_policy,
    fetch_gdelt_chunk,
    parse_gdelt_seendate,
    rows_to_feed_entries,
    to_gdelt_ts,
)


def test_parse_gdelt_seendate() -> None:
    dt = parse_gdelt_seendate("20260401120000")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 4 and dt.day == 1


def test_clip_range() -> None:
    now = datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 16, tzinfo=timezone.utc)
    a, b = clip_range_to_gdelt_policy(start, end, now=now)
    assert a >= now - timedelta(days=90)
    assert b == end  # end < now → unchanged (not forced to now)
    assert b <= now


def test_clip_range_empty_raises() -> None:
    now = datetime(2026, 4, 16, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        clip_range_to_gdelt_policy(now, now, now=now)


def test_rows_to_feed_entries() -> None:
    rows = [
        {
            "url": "https://example.com/a",
            "title": "Hello",
            "seendate": "20260401120000",
            "domain": "example.com",
            "snippet": "x",
        }
    ]
    fe = rows_to_feed_entries(rows)[0]
    assert isinstance(fe, FeedEntry)
    assert fe.link == "https://example.com/a"
    assert fe.source_id == "GDELT:example.com"


def test_fetch_gdelt_chunk_parses_json() -> None:
    fake_json = {
        "articles": [
            {
                "url": "https://x.test/1",
                "title": "T1",
                "seendate": "20260401120000",
                "domain": "x.test",
            }
        ]
    }
    raw = json.dumps(fake_json)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = raw
    mock_resp.json.return_value = fake_json
    mock_resp.raise_for_status = MagicMock()
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 2, tzinfo=timezone.utc)
    with patch("newscompare.gdelt_ingest.httpx.Client") as client_cls:
        cm = MagicMock()
        client_cls.return_value = cm
        cm.__enter__.return_value = cm
        cm.__exit__.return_value = None
        cm.get.return_value = mock_resp
        r = fetch_gdelt_chunk("q", start, end, maxrecords=250)
    assert len(r.entries) == 1
    assert r.entries[0].title == "T1"
    assert not r.truncated
    assert r.rate_limit_retries == 0


def test_to_gdelt_ts() -> None:
    dt = datetime(2026, 4, 1, 15, 30, 0, tzinfo=timezone.utc)
    assert to_gdelt_ts(dt) == "20260401153000"
