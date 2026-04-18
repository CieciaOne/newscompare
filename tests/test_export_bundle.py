"""Tests for analysis export."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from newscompare.export_bundle import export_for_analysis
from newscompare.storage import Storage, init_schema


def test_export_writes_bundle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "t.db")
        out = Path(tmp) / "out"
        storage = Storage(db_path)
        with storage.conn() as conn:
            init_schema(conn)
            now = datetime.now(timezone.utc)
            pub = (now - timedelta(days=5)).isoformat()
            conn.execute(
                """INSERT INTO articles (id, source_id, url, title, body, published_at, fetched_at)
                   VALUES (?,?,?,?,?,?,?)""",
                ("a1", "Src", "https://x/1", "T1", "body", pub, now.isoformat()),
            )
            conn.execute(
                "INSERT INTO claims (id, article_id, claim_text) VALUES (?,?,?)",
                ("c1", "a1", "claim one"),
            )
            conn.execute(
                "INSERT INTO topics (id, label, slug, created_at) VALUES (?,?,?,?)",
                ("t1", "Topic", "topic", now.isoformat()),
            )
            conn.execute(
                "INSERT INTO article_topics (article_id, topic_id) VALUES (?,?)",
                ("a1", "t1"),
            )

        with storage.conn() as conn:
            stats = export_for_analysis(conn, out, since_days=None)

        assert stats["article_count"] == 1
        assert stats["claim_count"] == 1
        assert (out / "stats.json").exists()
        assert (out / "articles.jsonl").exists()
        assert (out / "claims.csv").exists()
        assert (out / "topics.json").exists()
        loaded = json.loads((out / "stats.json").read_text(encoding="utf-8"))
        assert loaded["articles_by_source"]["Src"] == 1


def test_export_since_days_filters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "t.db")
        out = Path(tmp) / "out"
        storage = Storage(db_path)
        with storage.conn() as conn:
            init_schema(conn)
            now = datetime.now(timezone.utc)
            old_pub = (now - timedelta(days=100)).isoformat()
            new_pub = (now - timedelta(days=5)).isoformat()
            for aid, pub in [("old", old_pub), ("new", new_pub)]:
                conn.execute(
                    """INSERT INTO articles (id, source_id, url, title, body, published_at, fetched_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (aid, "S", f"https://x/{aid}", aid, "b", pub, now.isoformat()),
                )

        with storage.conn() as conn:
            stats = export_for_analysis(conn, out, since_days=30)

        assert stats["article_count"] == 1
        lines = (out / "articles.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["id"] == "new"
