"""Group articles into same-story clusters (time window + title similarity)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from newscompare.embeddings import embed_texts, cosine_similarity_matrix
from newscompare.storage import Storage
from newscompare.translation import content_for_compare


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def group_articles(
    storage: Storage,
    embedding_model: str,
    hours_window: int = 24,
    title_similarity_threshold: float = 0.5,
    *,
    source_ids: list[str] | None = None,
    max_articles: int | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Load articles from storage, group by time window then by title similarity.
    Returns list of groups; each group is a list of article dicts.

    source_ids: if set, only articles whose source_id is in this list (after time filter).
    max_articles: if set, cap how many recent articles are embedded/grouped (newest first).
    """
    with storage.conn() as conn:
        rows = conn.execute(
            "SELECT id, source_id, url, title, body, published_at, fetched_at, translated_title, translated_body FROM articles ORDER BY fetched_at DESC"
        ).fetchall()
    articles = [dict(r) for r in rows]
    if not articles:
        return []

    # Time window: use fetched_at or published_at
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours_window)
    def _article_time(a: dict) -> datetime:
        dt = _parse_dt(a.get("published_at")) or _parse_dt(a.get("fetched_at"))
        if dt is None:
            return now
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    recent = [a for a in articles if _article_time(a) >= cutoff]
    if not recent:
        recent = articles[: 50]

    if source_ids:
        allow = {s.strip() for s in source_ids if s.strip()}
        if allow:
            recent = [a for a in recent if (a.get("source_id") or "") in allow]
    if max_articles is not None and max_articles > 0:
        recent = recent[:max_articles]

    titles = [content_for_compare(a)[0] or "" for a in recent]
    embs = embed_texts(titles, model_name=embedding_model)
    sim = cosine_similarity_matrix(embs)

    # Simple clustering: greedy. Each article is in at most one group.
    n = len(recent)
    assigned = [False] * n
    groups: list[list[dict[str, Any]]] = []

    for i in range(n):
        if assigned[i]:
            continue
        group = [recent[i]]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if sim[i, j] >= title_similarity_threshold:
                group.append(recent[j])
                assigned[j] = True
        if len(group) >= 1:
            for a in group:
                a["title"] = content_for_compare(a)[0] or a.get("title") or ""
            groups.append(group)

    return groups
