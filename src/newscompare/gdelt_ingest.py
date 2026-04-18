"""Ingest article lists from the free GDELT DOC 2.0 API (date-limited rolling window)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from newscompare.feed_fetcher import FeedEntry

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_LOOKBACK_DAYS = 90
GDELT_MAXRECORDS_CAP = 250
DEFAULT_USER_AGENT = "NewsCompare/0.1 (GDELT DOC reader; research)"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_gdelt_ts(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y%m%d%H%M%S")


def parse_gdelt_seendate(value: Any) -> datetime | None:
    if value is None:
        return None
    s = str(value).strip()
    if len(s) < 14:
        return None
    s = s[:14]
    try:
        naive = datetime.strptime(s, "%Y%m%d%H%M%S")
        return naive.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def clip_range_to_gdelt_policy(
    start: datetime,
    end: datetime,
    *,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    now = now or utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    earliest = now - timedelta(days=GDELT_MAX_LOOKBACK_DAYS)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    if end > now:
        end = now
    if start < earliest:
        logger.warning("GDELT window: clipping start from %s to %s", start, earliest)
        start = earliest
    if start >= end:
        raise ValueError("Date range empty after clipping (start >= end). Check --start/--end.")
    return start, end


def _extract_article_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    arts = payload.get("articles")
    if isinstance(arts, list):
        return [x for x in arts if isinstance(x, dict)]
    for _k, v in payload.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            row0 = v[0]
            if "url" in row0 or "URL" in row0:
                return [x for x in v if isinstance(x, dict)]
    return []


def rows_to_feed_entries(rows: list[dict[str, Any]], *, source_prefix: str = "GDELT") -> list[FeedEntry]:
    out: list[FeedEntry] = []
    for r in rows:
        url = (r.get("url") or r.get("URL") or "").strip()
        title = (r.get("title") or r.get("Title") or "").strip()
        if not url or not title:
            continue
        domain = (r.get("domain") or r.get("Domain") or "").strip().lower()
        published = parse_gdelt_seendate(r.get("seendate") or r.get("seenDate") or r.get("date"))
        summary = r.get("snippet") or r.get("excerpt") or ""
        if not isinstance(summary, str):
            summary = str(summary) if summary else ""
        if len(summary) > 10000:
            summary = summary[:10000] + "..."
        source_id = f"{source_prefix}:{domain}" if domain else source_prefix
        out.append(
            FeedEntry(
                title=title,
                link=url,
                published=published,
                summary=summary if isinstance(summary, str) else "",
                source_id=source_id[:200],
            )
        )
    return out


@dataclass
class GdeltChunkResult:
    entries: list[FeedEntry]
    truncated: bool


def fetch_gdelt_chunk(
    query: str,
    start: datetime,
    end: datetime,
    *,
    maxrecords: int = GDELT_MAXRECORDS_CAP,
    sort: str = "datedesc",
    timeout: float = 45.0,
) -> GdeltChunkResult:
    maxrecords = min(max(1, maxrecords), GDELT_MAXRECORDS_CAP)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(maxrecords),
        "sort": sort,
        "startdatetime": to_gdelt_ts(start),
        "enddatetime": to_gdelt_ts(end),
    }
    url = f"{GDELT_DOC_API}?{urlencode(params)}"
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        text = resp.text.strip()
        if text.startswith("<") or "DOCTYPE html" in text[:200].upper():
            raise RuntimeError("GDELT returned HTML instead of JSON.")
        payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("GDELT response JSON root is not an object.")
    rows = _extract_article_dicts(payload)
    entries = rows_to_feed_entries(rows)
    truncated = len(rows) >= maxrecords
    return GdeltChunkResult(entries=entries, truncated=truncated)


def iter_gdelt_timerange(
    query: str,
    start: datetime,
    end: datetime,
    *,
    chunk_hours: int = 48,
    maxrecords: int = GDELT_MAXRECORDS_CAP,
    sort: str = "datedesc",
    sleep_seconds: float = 1.0,
    timeout: float = 45.0,
) -> tuple[list[FeedEntry], list[str]]:
    start, end = clip_range_to_gdelt_policy(start, end)
    if chunk_hours < 1:
        raise ValueError("chunk_hours must be >= 1")
    chunk = timedelta(hours=chunk_hours)
    warnings: list[str] = []
    seen_urls: set[str] = set()
    merged: list[FeedEntry] = []

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        try:
            result = fetch_gdelt_chunk(
                query,
                cursor,
                chunk_end,
                maxrecords=maxrecords,
                sort=sort,
                timeout=timeout,
            )
        except Exception as e:
            warnings.append(f"{to_gdelt_ts(cursor)}-{to_gdelt_ts(chunk_end)}: {e}")
            cursor = chunk_end
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        if result.truncated:
            warnings.append(
                f"{to_gdelt_ts(cursor)}-{to_gdelt_ts(chunk_end)}: hit maxrecords={maxrecords}; "
                "narrow --chunk-hours or refine query."
            )
        for fe in result.entries:
            if fe.link in seen_urls:
                continue
            seen_urls.add(fe.link)
            merged.append(fe)
        cursor = chunk_end
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return merged, warnings
