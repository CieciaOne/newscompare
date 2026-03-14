"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest

from newscompare.config import Config, ConfigError, FeedConfig


def test_from_dict_empty() -> None:
    c = Config.from_dict({})
    assert c.feeds == []
    assert c.database == "./newscompare.db"


def test_from_dict_feeds_strings() -> None:
    c = Config.from_dict({"feeds": ["https://a.com/feed", "https://b.com/feed"]})
    assert len(c.feeds) == 2
    assert c.feeds[0].url == "https://a.com/feed"
    assert c.feeds[0].source_id == "https://a.com/feed"


def test_from_dict_feeds_objects() -> None:
    c = Config.from_dict({
        "feeds": [
            {"url": "https://a.com/feed", "name": "A", "max_articles": 10},
        ]
    })
    assert len(c.feeds) == 1
    assert c.feeds[0].url == "https://a.com/feed"
    assert c.feeds[0].name == "A"
    assert c.feeds[0].source_id == "A"
    assert c.feeds[0].max_articles == 10


def test_load_missing_file_raises() -> None:
    with pytest.raises(ConfigError):
        Config.load("/nonexistent/config.yaml")


def test_load_from_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("feeds:\n  - url: https://x.com/feed\n    name: X\n")
        path = f.name
    try:
        c = Config.load(path)
        assert len(c.feeds) == 1
        assert c.feeds[0].name == "X"
    finally:
        Path(path).unlink(missing_ok=True)
