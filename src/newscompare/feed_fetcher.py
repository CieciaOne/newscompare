"""Fetch and parse RSS/Atom feeds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import feedparser
import httpx

from newscompare.config import FeedConfig
from newscompare.exceptions import FeedFetchError

logger = logging.getLogger(__name__)

USER_AGENT = "NewsCompare/0.1 (RSS reader; no auth)"


@dataclass
class FeedEntry:
    title: str
    link: str
    published: datetime | None
    summary: str
    source_id: str

    @classmethod
    def from_entry(cls, entry: Any, source_id: str) -> FeedEntry | None:
        title = getattr(entry, "title", None) or ""
        link = getattr(entry, "link", None) or ""
        if not title or not link:
            return None
        published = None
        if getattr(entry, "published_parsed", None):
            try:
                published = datetime(*entry.published_parsed[:6])
            except (TypeError, IndexError):
                pass
        summary = ""
        if getattr(entry, "summary", None):
            summary = entry.summary
        elif getattr(entry, "description", None):
            summary = entry.description
        if isinstance(summary, str) and len(summary) > 10000:
            summary = summary[:10000] + "..."
        return cls(title=title.strip(), link=link, published=published, summary=summary, source_id=source_id)


def fetch_feed(
    feed_config: FeedConfig,
    timeout: int = 10,
) -> list[FeedEntry]:
    """Fetch one feed and return list of entries. On failure logs and returns []."""
    logger.info("Fetching feed: %s (%s)", feed_config.source_id, feed_config.url[:60])
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"},
        ) as client:
            resp = client.get(feed_config.url)
            resp.raise_for_status()
            raw = resp.content
        # Pass content-type so feedparser can pick correct parser (e.g. Atom)
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        parsed = feedparser.parse(raw, response_headers={"content-type": content_type or "application/xml"})
    except httpx.HTTPError as e:
        logger.warning("Feed fetch failed %s: %s", feed_config.url, e)
        raise FeedFetchError(f"Failed to fetch {feed_config.url}: {e}") from e
    except Exception as e:
        logger.warning("Feed fetch failed %s: %s", feed_config.url, e)
        raise FeedFetchError(f"Failed to fetch {feed_config.url}: {e}") from e

    if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
        hint = ""
        if raw.lstrip()[:50].lower().startswith(b"<!doctype") or raw.lstrip().startswith(b"<html"):
            hint = " (server returned HTML, not RSS/Atom — feed URL may be wrong or deprecated)"
        raise FeedFetchError(f"Invalid feed or parse error: {feed_config.url}{hint}")

    entries = getattr(parsed, "entries", []) or []
    source_id = feed_config.source_id
    out: list[FeedEntry] = []
    for entry in entries[: feed_config.max_articles]:
        fe = FeedEntry.from_entry(entry, source_id)
        if fe:
            out.append(fe)
    logger.info("Feed %s: %d entries", source_id, len(out))
    return out


def fetch_all_feeds(
    feed_configs: list[FeedConfig],
    timeout: int = 10,
) -> list[FeedEntry]:
    """Fetch all feeds; on per-feed failure log and continue. Returns combined list."""
    all_entries: list[FeedEntry] = []
    for fc in feed_configs:
        try:
            all_entries.extend(fetch_feed(fc, timeout=timeout))
        except FeedFetchError as e:
            logger.warning("Skipping feed after error: %s", e)
    return all_entries
