"""Ingest article lists from the free GDELT DOC 2.0 API (date-limited rolling window)."""

from __future__ import annotations

import logging
import math
import random
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
        tone = r.get("tone", r.get("Tone"))
        if tone is not None and str(tone).strip() != "":
            summary = f"[GDELT_tone:{tone}] {summary}".strip()
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
    #: How many 429/503 responses were seen before a successful response for this chunk.
    rate_limit_retries: int = 0


def _gdelt_retry_after_seconds(header_val: str | None, attempt: int, *, is_429: bool) -> float:
    if header_val:
        s = str(header_val).strip()
        if s.isdigit():
            return float(s)
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(s)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                delta = (dt.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
                if delta > 0:
                    return min(600.0, delta)
        except (TypeError, ValueError, OSError):
            pass
    base = 3.0 if is_429 else 2.0
    return min(240.0, (2**attempt) * base + random.uniform(0, 2.5))


def _fetch_gdelt_chunk_with_client(
    client: httpx.Client,
    query: str,
    start: datetime,
    end: datetime,
    *,
    maxrecords: int,
    sort: str,
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
    rate_limit_retries = 0
    resp: httpx.Response | None = None
    for attempt in range(8):
        resp = client.get(url)
        if resp.status_code == 429 or resp.status_code >= 503:
            rate_limit_retries += 1
            is_429 = resp.status_code == 429
            ra = resp.headers.get("Retry-After")
            wait_s = _gdelt_retry_after_seconds(ra, attempt, is_429=is_429)
            logger.warning(
                "GDELT HTTP %s, sleeping %.1fs (retry %d/8) chunk end=%s",
                resp.status_code,
                wait_s,
                attempt + 1,
                to_gdelt_ts(end),
            )
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(
            f"GDELT rate limited or unavailable after 8 retries (last status {getattr(resp, 'status_code', '?')})"
        )
    text = resp.text.strip()
    if text.startswith("<") or "DOCTYPE html" in text[:200].upper():
        raise RuntimeError("GDELT returned HTML instead of JSON.")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("GDELT response JSON root is not an object.")
    rows = _extract_article_dicts(payload)
    entries = rows_to_feed_entries(rows)
    truncated = len(rows) >= maxrecords
    return GdeltChunkResult(
        entries=entries,
        truncated=truncated,
        rate_limit_retries=rate_limit_retries,
    )


def fetch_gdelt_chunk(
    query: str,
    start: datetime,
    end: datetime,
    *,
    maxrecords: int = GDELT_MAXRECORDS_CAP,
    sort: str = "datedesc",
    timeout: float = 45.0,
) -> GdeltChunkResult:
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        return _fetch_gdelt_chunk_with_client(
            client, query, start, end, maxrecords=maxrecords, sort=sort
        )


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
    verbose: bool = True,
) -> tuple[list[FeedEntry], list[str]]:
    start, end = clip_range_to_gdelt_policy(start, end)
    if chunk_hours < 1:
        raise ValueError("chunk_hours must be >= 1")
    chunk = timedelta(hours=chunk_hours)
    span_sec = max(0.0, (end - start).total_seconds())
    chunk_sec = max(1.0, chunk.total_seconds())
    total_chunks = max(1, math.ceil(span_sec / chunk_sec))
    warnings: list[str] = []
    seen_urls: set[str] = set()
    merged: list[FeedEntry] = []

    if verbose:
        logger.info(
            "GDELT range UTC %s → %s (~%d chunks of %dh, maxrecords=%d, query=%r)",
            to_gdelt_ts(start),
            to_gdelt_ts(end),
            total_chunks,
            chunk_hours,
            maxrecords,
            query[:120] + ("…" if len(query) > 120 else ""),
        )

    chunk_idx = 0
    cursor = start
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    # Minimum wall time between *starts* of consecutive chunk requests (GDELT often throttles
    # the 3rd+ hit in a burst when chunks 1–2 return quickly and --sleep only ran after each).
    prev_outer_start: float | None = None
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as http_client:
        while cursor < end:
            chunk_idx += 1
            chunk_end = min(cursor + chunk, end)
            if prev_outer_start is not None and sleep_seconds > 0:
                min_next = prev_outer_start + sleep_seconds * (1.0 + random.uniform(0.0, 0.12))
                delay = min_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            t_outer = time.perf_counter()
            try:
                result = _fetch_gdelt_chunk_with_client(
                    http_client,
                    query,
                    cursor,
                    chunk_end,
                    maxrecords=maxrecords,
                    sort=sort,
                )
            except Exception as e:
                msg = f"{to_gdelt_ts(cursor)}-{to_gdelt_ts(chunk_end)}: {e}"
                warnings.append(msg)
                if verbose:
                    logger.warning(
                        "GDELT chunk %d/%d %s → %s FAILED: %s",
                        chunk_idx,
                        total_chunks,
                        to_gdelt_ts(cursor),
                        to_gdelt_ts(chunk_end),
                        e,
                    )
                prev_outer_start = t_outer
                cursor = chunk_end
                continue

            before = len(merged)
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
            new_in_chunk = len(merged) - before
            elapsed = time.perf_counter() - t_outer
            if result.rate_limit_retries > 0:
                cool = min(180.0, 10.0 * result.rate_limit_retries + random.uniform(0, 5.0))
                if verbose:
                    logger.info(
                        "GDELT extra cooldown %.1fs after %d rate-limit hit(s) before next chunk",
                        cool,
                        result.rate_limit_retries,
                    )
                time.sleep(cool)
            if verbose:
                logger.info(
                    "GDELT chunk %d/%d %s → %s: returned=%d new_unique=%d total_unique=%d truncated=%s (%.1fs)",
                    chunk_idx,
                    total_chunks,
                    to_gdelt_ts(cursor),
                    to_gdelt_ts(chunk_end),
                    len(result.entries),
                    new_in_chunk,
                    len(merged),
                    result.truncated,
                    elapsed,
                )
            if result.truncated and verbose:
                logger.warning(
                    "GDELT chunk %d/%d capped at maxrecords=%d — consider smaller --chunk-hours for this query.",
                    chunk_idx,
                    total_chunks,
                    maxrecords,
                )
            prev_outer_start = t_outer
            cursor = chunk_end

    if verbose:
        logger.info("GDELT done: %d unique URLs after merge (%d warning lines)", len(merged), len(warnings))

    return merged, warnings
