"""Batch topic extraction: cluster articles, label clusters with LLM, assign to DB."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from newscompare.embeddings import embed_texts
from newscompare.config import Config, LLMConfig
from newscompare.storage import Storage, save_topics, save_article_topics
from newscompare.translation import content_for_compare

logger = logging.getLogger(__name__)

TOPIC_LABEL_PROMPT = """You are given a list of news headlines that belong to the same topic. Output exactly one short topic label (2-6 words) that describes the common theme. Use title case. Output only the label, nothing else.
Headlines:
"""


def _slugify(label: str) -> str:
    s = re.sub(r"[^\w\s-]", "", label.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s or "topic"


def _label_cluster_ollama(titles: list[str], config: LLMConfig) -> str:
    """Use Ollama to produce one short topic label from a list of headlines."""
    try:
        import ollama
    except ImportError:
        return "Uncategorized"
    sample = "\n".join(titles[:15])
    prompt = TOPIC_LABEL_PROMPT + sample
    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 50},
        )
        content = (response.get("message") or {}).get("content") or ""
        label = content.strip().split("\n")[0].strip()[:80]
        return label or "Uncategorized"
    except Exception as e:
        logger.warning("Ollama topic label failed: %s", e)
        return "Uncategorized"


def _label_cluster(titles: list[str], config: LLMConfig) -> str:
    if config.provider == "ollama":
        return _label_cluster_ollama(titles, config)
    return "Uncategorized"


def extract_topics(
    storage: Storage,
    config: Config,
    hours_window: int = 168,  # 7 days
    max_topics: int = 25,
    min_articles_per_topic: int = 2,
) -> list[dict[str, Any]]:
    """
    Load articles in time window, cluster by embedding similarity, label each cluster with LLM,
    save topics and assignments to DB. Returns list of {id, label, slug, article_count}.
    """
    with storage.conn() as conn:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_window)).isoformat()
        rows = conn.execute(
            """SELECT id, source_id, title, body, published_at, fetched_at, translated_title, translated_body
               FROM articles WHERE COALESCE(published_at, fetched_at) >= ?
               ORDER BY fetched_at DESC""",
            (cutoff,),
        ).fetchall()
    articles = [dict(r) for r in rows]
    logger.info("Topic extraction: %d articles in window (last %d hours)", len(articles), hours_window)
    if len(articles) < min_articles_per_topic:
        logger.warning("Not enough articles for topic extraction (need at least %d)", min_articles_per_topic)
        return []

    # Embed title + short body (use translated when present so Polish etc. cluster with English)
    texts = []
    for a in articles:
        title, body = content_for_compare(a)
        t = (title or "") + " " + ((body or "")[:500])
        texts.append(t.strip())
    embs = embed_texts(texts, model_name=config.embedding_model)
    n = len(articles)

    # Hierarchical clustering; cut to get ~max_topics clusters
    if n <= 1:
        return []
    cond = pdist(embs, metric="cosine")
    Z = linkage(cond, method="average")
    # fcluster: t threshold; smaller t = more clusters. We want roughly max_topics.
    # max_d is max distance within cluster. Try a few values.
    for t in [0.5, 0.7, 0.9, 1.0, 1.2]:
        labels = fcluster(Z, t=t, criterion="distance")
        n_clusters = len(set(labels))
        if 2 <= n_clusters <= max_topics + 5:
            break
    else:
        labels = fcluster(Z, t=1.0, criterion="distance")

    # Build clusters: cluster_id -> list of (article, title)
    clusters: dict[int, list[tuple[dict, str]]] = {}
    for i, art in enumerate(articles):
        cid = int(labels[i])
        display_title = content_for_compare(art)[0] or texts[i][:200]
        clusters.setdefault(cid, []).append((art, display_title[:200]))

    # Filter small clusters; assign each article to one topic (its cluster)
    topic_assignments: dict[str, list[str]] = {}  # article_id -> [topic_id]
    topics_raw: list[tuple[str, str, str]] = []   # (id, label, slug)
    seen_slugs: set[str] = set()

    for cid, members in clusters.items():
        if len(members) < min_articles_per_topic:
            continue
        titles = [m[1] for m in members]
        label = _label_cluster(titles, config.llm)
        slug = _slugify(label)
        if slug in seen_slugs:
            slug = f"{slug}-{cid}"
        seen_slugs.add(slug)
        tid = str(uuid.uuid4())
        topics_raw.append((tid, label, slug))
        for art, _ in members:
            topic_assignments.setdefault(art["id"], []).append(tid)

    if not topics_raw:
        logger.warning("No clusters with >= %d articles after grouping", min_articles_per_topic)
        return []

    logger.info("Labeling %d clusters with LLM", len(topics_raw))
    with storage.conn() as conn:
        save_topics(conn, topics_raw)
        for aid, tids in topic_assignments.items():
            save_article_topics(conn, aid, tids)

    # Return with counts
    result = []
    with storage.conn() as conn:
        for tid, label, slug in topics_raw:
            count = sum(1 for aids in topic_assignments.values() if tid in aids)
            result.append({"id": tid, "label": label, "slug": slug, "article_count": count})
    return result
