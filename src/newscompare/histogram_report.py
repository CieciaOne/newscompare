"""Day-level article histogram and calendar-gap detection (read-only on DB)."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any


def _day_from_row(published_at: str | None, fetched_at: str | None) -> str | None:
    ts = published_at or fetched_at or ""
    if isinstance(ts, str) and len(ts) >= 10:
        return ts[:10]
    return None


def day_histogram_from_conn(
    conn: Any,
    *,
    source_like: str | None = None,
) -> tuple[Counter[str], int]:
    """
    Return (Counter by YYYY-MM-DD, total rows used).
    source_like: optional SQL LIKE pattern, e.g. 'GDELT:%' or '%'.
    """
    q = """SELECT published_at, fetched_at FROM articles WHERE 1=1"""
    params: list[Any] = []
    if source_like:
        q += " AND source_id LIKE ?"
        params.append(source_like)
    rows = conn.execute(q, params).fetchall()
    by_day: Counter[str] = Counter()
    used = 0
    for r in rows:
        row = dict(r) if hasattr(r, "keys") else {"published_at": r[0], "fetched_at": r[1]}
        d = _day_from_row(row.get("published_at"), row.get("fetched_at"))
        if d and d != "unknown":
            by_day[d] += 1
            used += 1
    return by_day, used


def calendar_gaps(by_day: Counter[str]) -> tuple[list[str], str | None, str | None, int]:
    """
    Days in [min_day, max_day] inclusive with zero articles.
    Returns (gap_days_sorted, first_day, last_day, span_inclusive_days).
    """
    if not by_day:
        return [], None, None, 0
    days_sorted = sorted(by_day.keys())
    first_s, last_s = days_sorted[0], days_sorted[-1]
    d0 = date.fromisoformat(first_s)
    d1 = date.fromisoformat(last_s)
    span = (d1 - d0).days + 1
    known = set(by_day.keys())
    gaps: list[str] = []
    cur = d0
    while cur <= d1:
        s = cur.isoformat()
        if s not in known:
            gaps.append(s)
        cur += timedelta(days=1)
    return gaps, first_s, last_s, span


def ascii_bars(by_day: Counter[str], *, width: int = 50, max_lines: int = 200) -> str:
    """Horizontal bars: one line per day, newest first, capped for terminal."""
    if not by_day:
        return "(no dated articles)\n"
    items = sorted(by_day.items(), key=lambda x: x[0], reverse=True)
    mx = max(by_day.values()) or 1
    cap = max(1, max_lines)
    lines: list[str] = []
    for day, n in items[:cap]:
        bar_len = max(1, int(round((n / mx) * width)))
        lines.append(f"{day}  {n:5d}  {'█' * bar_len}")
    if len(items) > cap:
        lines.append(f"... ({len(items) - cap} more days not shown)")
    return "\n".join(lines) + "\n"


def build_report_dict(by_day: Counter[str], *, used_rows: int) -> dict[str, Any]:
    gaps, first_d, last_d, span = calendar_gaps(by_day)
    return {
        "total_articles_with_date": used_rows,
        "distinct_days_with_articles": len(by_day),
        "first_day": first_d,
        "last_day": last_d,
        "calendar_span_days_inclusive": span,
        "gap_days_count": len(gaps),
        "gap_days": gaps[:500],
        "gap_days_truncated": len(gaps) > 500,
        "articles_by_day": dict(sorted(by_day.items())),
    }
