"""Tests for day histogram and calendar gaps."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from newscompare.histogram_report import (
    build_report_dict,
    calendar_gaps,
    day_histogram_from_conn,
)
from newscompare.storage import Storage, init_schema


def test_calendar_gaps_finds_missing_days() -> None:
    c = Counter({"2024-01-01": 1, "2024-01-03": 2})
    gaps, first, last, span = calendar_gaps(c)
    assert first == "2024-01-01"
    assert last == "2024-01-03"
    assert span == 3
    assert gaps == ["2024-01-02"]


def test_calendar_gaps_empty() -> None:
    gaps, first, last, span = calendar_gaps(Counter())
    assert gaps == []
    assert first is None
    assert last is None
    assert span == 0


def test_day_histogram_and_source_like() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "h.db")
        storage = Storage(db_path)
        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        d1 = (now - timedelta(days=1)).date().isoformat()
        d0 = now.date().isoformat()
        with storage.conn() as conn:
            init_schema(conn)
            for aid, src, pub in [
                ("a1", "GDELT:x", f"{d1}T10:00:00+00:00"),
                ("a2", "RSS:y", f"{d1}T11:00:00+00:00"),
                ("a3", "GDELT:y", f"{d0}T08:00:00+00:00"),
            ]:
                conn.execute(
                    """INSERT INTO articles (id, source_id, url, title, body, published_at, fetched_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (aid, src, f"https://x/{aid}", aid, "b", pub, now.isoformat()),
                )

        with storage.conn() as conn:
            all_days, used = day_histogram_from_conn(conn)
            gdelt_days, gdelt_used = day_histogram_from_conn(conn, source_like="GDELT:%")

        assert used == 3
        assert all_days[d1] == 2
        assert all_days[d0] == 1
        assert gdelt_used == 2
        assert gdelt_days[d1] == 1
        assert gdelt_days[d0] == 1

        rep = build_report_dict(all_days, used_rows=used)
        assert rep["total_articles_with_date"] == 3
        assert rep["gap_days_count"] == 0  # consecutive calendar range from d1 to d0? d1 and d0 - if d0 is next day of d1 then no gap. d0 = 2024-06-15, d1 = 2024-06-14 - consecutive, no gap.


def test_build_report_dict_gap_span() -> None:
    c = Counter({"2024-01-01": 1, "2024-01-05": 1})
    rep = build_report_dict(c, used_rows=2)
    assert rep["calendar_span_days_inclusive"] == 5
    assert rep["gap_days_count"] == 3
    assert set(rep["gap_days"]) == {"2024-01-02", "2024-01-03", "2024-01-04"}
