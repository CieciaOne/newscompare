"""SQLite storage for articles and extracted claims."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

from newscompare.feed_fetcher import FeedEntry

logger = logging.getLogger(__name__)


def _adapt_dt(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _convert_dt(s: bytes | str | None) -> datetime | None:
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s.decode() if isinstance(s, bytes) else s)
    except (ValueError, TypeError):
        return None


sqlite3.register_adapter(datetime, _adapt_dt)
sqlite3.register_converter("datetime", _convert_dt)


@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they do not exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            published_at TEXT,
            fetched_at TEXT NOT NULL,
            translated_title TEXT,
            translated_body TEXT
        );
    """)
    # Add translated columns if table already existed without them
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN translated_title TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN translated_body TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN story_summary TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN story_incident_json TEXT")
    except sqlite3.OperationalError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            article_id TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        );
        CREATE TABLE IF NOT EXISTS topics (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS article_topics (
            article_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            PRIMARY KEY (article_id, topic_id),
            FOREIGN KEY (article_id) REFERENCES articles(id),
            FOREIGN KEY (topic_id) REFERENCES topics(id)
        );
        CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id);
        CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
        CREATE INDEX IF NOT EXISTS idx_claims_article ON claims(article_id);
        CREATE INDEX IF NOT EXISTS idx_article_topics_topic ON article_topics(topic_id);
        CREATE TABLE IF NOT EXISTS topic_compare_reports (
            topic_id TEXT NOT NULL,
            sources_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            claims_json TEXT NOT NULL,
            articles_json TEXT NOT NULL,
            PRIMARY KEY (topic_id, sources_key)
        );
    """)


def insert_article(conn: sqlite3.Connection, entry: FeedEntry, body: str) -> str:
    """Insert one article; body is title+summary or extracted text. Returns id."""
    aid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    pub = entry.published.isoformat() if entry.published else None
    conn.execute(
        "INSERT INTO articles (id, source_id, url, title, body, published_at, fetched_at) VALUES (?,?,?,?,?,?,?)",
        (aid, entry.source_id, entry.link, entry.title, body or entry.summary, pub, now),
    )
    return aid


def update_article_body(conn: sqlite3.Connection, article_id: str, body: str) -> None:
    """Update body for an existing article."""
    conn.execute("UPDATE articles SET body = ? WHERE id = ?", (body, article_id))


def update_article_translation(
    conn: sqlite3.Connection,
    article_id: str,
    translated_title: str | None,
    translated_body: str | None,
) -> None:
    """Store English translation for an article (original title/body kept)."""
    conn.execute(
        "UPDATE articles SET translated_title = ?, translated_body = ? WHERE id = ?",
        (translated_title or "", translated_body or "", article_id),
    )


def get_article_by_url(conn: sqlite3.Connection, url: str) -> dict[str, Any] | None:
    """Return article row as dict if exists."""
    row = conn.execute("SELECT * FROM articles WHERE url = ?", (url,)).fetchone()
    return dict(row) if row else None


def get_article_by_id(conn: sqlite3.Connection, article_id: str) -> dict[str, Any] | None:
    """Return article row as dict if exists."""
    row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    return dict(row) if row else None


def list_articles(
    conn: sqlite3.Connection,
    source_id: str | None = None,
    since: datetime | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """List articles optionally filtered by source and since date."""
    q = "SELECT * FROM articles WHERE 1=1"
    params: list[Any] = []
    if source_id:
        q += " AND source_id = ?"
        params.append(source_id)
    if since:
        q += " AND fetched_at >= ?"
        params.append(since.isoformat())
    q += " ORDER BY fetched_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def list_articles_by_day(
    conn: sqlite3.Connection,
    days: int = 30,
    limit_per_day: int = 200,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return articles grouped by day (date string YYYY-MM-DD), newest first. Uses fetched_at or published_at for date."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT id, source_id, url, title, published_at, fetched_at, translated_title, translated_body
           FROM articles WHERE COALESCE(published_at, fetched_at) >= ?
           ORDER BY COALESCE(published_at, fetched_at) DESC""",
        (cutoff,),
    ).fetchall()
    articles = [dict(r) for r in rows]
    by_day: dict[str, list[dict[str, Any]]] = {}
    for a in articles:
        ts = a.get("published_at") or a.get("fetched_at") or ""
        day = ts[:10] if len(ts) >= 10 else "unknown"
        by_day.setdefault(day, []).append(a)
    # Sort days descending, cap per day
    sorted_days = sorted(by_day.keys(), reverse=True)
    return [(d, by_day[d][:limit_per_day]) for d in sorted_days]


def list_articles_needing_translation(
    conn: sqlite3.Connection,
    since: datetime | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """List articles where translated_body IS NULL (not yet processed by translate command)."""
    q = "SELECT * FROM articles WHERE translated_body IS NULL"
    params: list[Any] = []
    if since:
        q += " AND fetched_at >= ?"
        params.append(since.isoformat())
    q += " ORDER BY fetched_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def save_story_summary(conn: sqlite3.Connection, article_id: str, summary: str | None) -> None:
    """Set story_summary for an article. Summary should be a short narrative (2–4 sentences)."""
    conn.execute("UPDATE articles SET story_summary = ? WHERE id = ?", (summary or "", article_id))


def save_story_incident(conn: sqlite3.Connection, article_id: str, incident_json: str | None) -> None:
    """Store canonical incident object as JSON string (from structured LLM extract)."""
    conn.execute("UPDATE articles SET story_incident_json = ? WHERE id = ?", (incident_json or "", article_id))


def save_claims(conn: sqlite3.Connection, article_id: str, claims: list[str]) -> None:
    """Replace all claims for an article with the given list. Claim text is normalized before storage."""
    from newscompare.claims_util import normalize_claims

    conn.execute("DELETE FROM claims WHERE article_id = ?", (article_id,))
    for text in normalize_claims(claims):
        cid = str(uuid.uuid4())
        conn.execute("INSERT INTO claims (id, article_id, claim_text) VALUES (?,?,?)", (cid, article_id, text))


def get_claims_for_article(conn: sqlite3.Connection, article_id: str) -> list[str]:
    """Return list of claim texts for an article."""
    rows = conn.execute("SELECT claim_text FROM claims WHERE article_id = ? ORDER BY id", (article_id,)).fetchall()
    return [r[0] for r in rows]


def get_articles_with_claims(conn: sqlite3.Connection, article_ids: list[str]) -> dict[str, list[str]]:
    """Return {article_id: [claim_text, ...]} for given ids."""
    out: dict[str, list[str]] = {aid: [] for aid in article_ids}
    if not article_ids:
        return out
    placeholders = ",".join("?" * len(article_ids))
    rows = conn.execute(
        f"SELECT article_id, claim_text FROM claims WHERE article_id IN ({placeholders}) ORDER BY article_id, id",
        article_ids,
    ).fetchall()
    for r in rows:
        out[r[0]].append(r[1])
    return out


def save_topics(conn: sqlite3.Connection, topics: list[tuple[str, str, str]]) -> None:
    """Save topics: list of (id, label, slug). Replaces all existing topics and article_topics."""
    conn.execute("DELETE FROM article_topics")
    conn.execute("DELETE FROM topics")
    now = datetime.utcnow().isoformat()
    for tid, label, slug in topics:
        conn.execute(
            "INSERT INTO topics (id, label, slug, created_at) VALUES (?, ?, ?, ?)",
            (tid, label, slug, now),
        )


def save_article_topics(conn: sqlite3.Connection, article_id: str, topic_ids: list[str]) -> None:
    """Assign topic_ids to article_id."""
    conn.execute("DELETE FROM article_topics WHERE article_id = ?", (article_id,))
    for tid in topic_ids:
        conn.execute("INSERT INTO article_topics (article_id, topic_id) VALUES (?, ?)", (article_id, tid))


def list_topics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """List all topics with article count."""
    rows = conn.execute(
        """SELECT t.id, t.label, t.slug, t.created_at, COUNT(at.article_id) AS article_count
           FROM topics t
           LEFT JOIN article_topics at ON t.id = at.topic_id
           GROUP BY t.id ORDER BY article_count DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_articles_by_topic(
    conn: sqlite3.Connection,
    topic_id: str,
    order_by_time: bool = True,
) -> list[dict[str, Any]]:
    """Return articles assigned to this topic, with source_id, title, url, published_at, fetched_at."""
    q = """
        SELECT a.id, a.source_id, a.url, a.title, a.published_at, a.fetched_at, a.translated_title, a.translated_body
        FROM articles a
        INNER JOIN article_topics at ON a.id = at.article_id
        WHERE at.topic_id = ?
    """
    if order_by_time:
        q += " ORDER BY COALESCE(a.published_at, a.fetched_at) ASC"
    else:
        q += " ORDER BY a.fetched_at DESC"
    rows = conn.execute(q, (topic_id,)).fetchall()
    return [dict(r) for r in rows]


def get_topic(conn: sqlite3.Connection, topic_id: str) -> dict[str, Any] | None:
    """Return single topic by id or slug."""
    row = conn.execute("SELECT id, label, slug, created_at FROM topics WHERE id = ? OR slug = ?", (topic_id, topic_id)).fetchone()
    return dict(row) if row else None


def save_topic_compare_report(
    conn: sqlite3.Connection,
    topic_id: str,
    sources_key: str,
    claims: list[dict[str, Any]],
    articles: list[dict[str, Any]],
) -> None:
    """Store or replace comparison report for (topic_id, sources_key)."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO topic_compare_reports (topic_id, sources_key, created_at, claims_json, articles_json)
           VALUES (?, ?, ?, ?, ?)""",
        (topic_id, sources_key, now, json.dumps(claims), json.dumps(articles)),
    )


def get_topic_compare_report(
    conn: sqlite3.Connection,
    topic_id: str,
    sources_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return (claims, articles) if a report exists, else None."""
    row = conn.execute(
        "SELECT claims_json, articles_json FROM topic_compare_reports WHERE topic_id = ? AND sources_key = ?",
        (topic_id, sources_key),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0]), json.loads(row[1])
    except (json.JSONDecodeError, TypeError):
        return None


def list_reports(
    conn: sqlite3.Connection,
    topic_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List saved comparison reports, optionally for one topic. Newest first."""
    if topic_id:
        rows = conn.execute(
            """SELECT r.topic_id, r.sources_key, r.created_at, t.label
               FROM topic_compare_reports r
               LEFT JOIN topics t ON r.topic_id = t.id
               WHERE r.topic_id = ?
               ORDER BY r.created_at DESC
               LIMIT ?""",
            (topic_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT r.topic_id, r.sources_key, r.created_at, t.label
               FROM topic_compare_reports r
               LEFT JOIN topics t ON r.topic_id = t.id
               ORDER BY r.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(zip(["topic_id", "sources_key", "created_at", "topic_label"], r)) for r in rows]


class Storage:
    """Thin wrapper around DB path and connection."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def conn(self) -> Generator[sqlite3.Connection, None, None]:
        with _connect(self.db_path) as c:
            init_schema(c)
            yield c
