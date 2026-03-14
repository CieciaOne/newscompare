"""Tests for feed fetcher (parsing)."""

import pytest

import feedparser

from newscompare.feed_fetcher import FeedEntry


def test_feed_entry_from_entry_minimal() -> None:
    class E:
        title = "Hello"
        link = "https://example.com/1"
        published_parsed = None
        summary = ""

    entry = FeedEntry.from_entry(E(), "TestSource")
    assert entry is not None
    assert entry.title == "Hello"
    assert entry.link == "https://example.com/1"
    assert entry.source_id == "TestSource"
    assert entry.published is None
    assert entry.summary == ""


def test_feed_entry_from_entry_missing_title_returns_none() -> None:
    class E:
        title = ""
        link = "https://example.com/1"
        published_parsed = None

    assert FeedEntry.from_entry(E(), "S") is None


def test_feed_entry_from_entry_missing_link_returns_none() -> None:
    class E:
        title = "Hi"
        link = ""
        published_parsed = None

    assert FeedEntry.from_entry(E(), "S") is None


def test_feed_parsed_with_fixture() -> None:
    """Parse a minimal RSS XML string."""
    rss = b"""<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Test</title>
        <item>
          <title>Item 1</title>
          <link>https://example.com/1</link>
          <description>Summary 1</description>
        </item>
      </channel>
    </rss>
    """
    parsed = feedparser.parse(rss)
    assert parsed.entries
    entry = parsed.entries[0]
    fe = FeedEntry.from_entry(entry, "Fixture")
    assert fe is not None
    assert fe.title == "Item 1"
    assert fe.link == "https://example.com/1"
    assert "Summary" in fe.summary or fe.summary == ""
