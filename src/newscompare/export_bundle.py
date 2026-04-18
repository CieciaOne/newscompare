"""Export articles, claims, topics, and aggregate stats for offline analysis."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _article_effective_time(row: dict[str, Any]) -> datetime | None:
    return _parse_ts(row.get("published_at")) or _parse_ts(row.get("fetched_at"))


def _since_cutoff_iso(since_days: int | None) -> str | None:
    if since_days is None or since_days <= 0:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    return cutoff.isoformat()


def export_for_analysis(
    conn: Any,
    out_dir: Path,
    *,
    since_days: int | None = None,
) -> dict[str, Any]:
    """
    Write analysis bundle under out_dir:
      stats.json, articles.jsonl, claims.csv, topics.json
    Returns the stats dict written to stats.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff_iso = _since_cutoff_iso(since_days)

    where = "1=1"
    params: list[Any] = []
    if cutoff_iso:
        where = "COALESCE(published_at, fetched_at) >= ?"
        params.append(cutoff_iso)

    rows = conn.execute(
        f"""SELECT id, source_id, url, title, published_at, fetched_at,
                   LENGTH(COALESCE(body,'')) AS body_chars,
                   LENGTH(COALESCE(story_summary,'')) AS story_chars,
                   (SELECT COUNT(*) FROM claims c WHERE c.article_id = articles.id) AS claim_count
            FROM articles WHERE {where}
            ORDER BY COALESCE(published_at, fetched_at) DESC""",
        params,
    ).fetchall()
    articles = [dict(r) for r in rows]

    times = [_article_effective_time(a) for a in articles]
    times_valid = [t for t in times if t is not None]
    min_t = min(times_valid).isoformat() if times_valid else None
    max_t = max(times_valid).isoformat() if times_valid else None

    by_source = Counter(a["source_id"] for a in articles)
    by_day: Counter[str] = Counter()
    for a in articles:
        ts = a.get("published_at") or a.get("fetched_at") or ""
        day = ts[:10] if isinstance(ts, str) and len(ts) >= 10 else "unknown"
        by_day[day] += 1

    if cutoff_iso:
        claim_sql = """SELECT c.id AS claim_id, c.article_id, a.source_id, c.claim_text
            FROM claims c
            INNER JOIN articles a ON a.id = c.article_id
            WHERE COALESCE(a.published_at, a.fetched_at) >= ?"""
        claim_params: list[Any] = [cutoff_iso]
    else:
        claim_sql = """SELECT c.id AS claim_id, c.article_id, a.source_id, c.claim_text
            FROM claims c
            INNER JOIN articles a ON a.id = c.article_id"""
        claim_params = []
    claim_rows = conn.execute(claim_sql, claim_params).fetchall()

    topic_rows = conn.execute(
        """SELECT t.id, t.label, t.slug, t.created_at, COUNT(at.article_id) AS article_count
           FROM topics t
           LEFT JOIN article_topics at ON t.id = at.topic_id
           GROUP BY t.id ORDER BY article_count DESC""",
    ).fetchall()
    topics = [dict(r) for r in topic_rows]

    reports_n = conn.execute("SELECT COUNT(*) FROM topic_compare_reports").fetchone()[0]

    stats: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_since_days": since_days,
        "article_count": len(articles),
        "claim_count": len(claim_rows),
        "topic_count": len(topics),
        "topic_compare_reports_count": reports_n,
        "date_range_effective_utc": {"min": min_t, "max": max_t},
        "articles_by_source": dict(by_source.most_common()),
        "articles_by_day": dict(sorted(by_day.items(), reverse=True)[:120]),
    }

    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", stats_path)

    articles_path = out_dir / "articles.jsonl"
    with open(articles_path, "w", encoding="utf-8") as f:
        for a in articles:
            line = {
                "id": a["id"],
                "source_id": a["source_id"],
                "url": a["url"],
                "title": a["title"],
                "published_at": a.get("published_at"),
                "fetched_at": a.get("fetched_at"),
                "body_chars": a.get("body_chars"),
                "story_chars": a.get("story_chars"),
                "claim_count": a.get("claim_count"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d rows)", articles_path, len(articles))

    claims_path = out_dir / "claims.csv"
    with open(claims_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "article_id", "source_id", "claim_text"])
        for r in claim_rows:
            w.writerow([r["claim_id"], r["article_id"], r["source_id"], r["claim_text"]])
    logger.info("Wrote %s (%d rows)", claims_path, len(claim_rows))

    topics_path = out_dir / "topics.json"
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", topics_path)

    return stats
