"""Batch topic extraction: cluster articles, label clusters with LLM, assign to DB."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from newscompare.embeddings import cosine_similarity_matrix, embed_texts
from newscompare.config import Config, LLMConfig
from newscompare.llm_dataset import _extract_json_object
from newscompare.storage import Storage, save_topics, save_article_topics
from newscompare.topic_verdict import classify_topic_cluster_ollama, final_topic_label
from newscompare.translation import content_for_compare

logger = logging.getLogger(__name__)

TOPIC_LABEL_PROMPT = """Headlines below belong to one cluster. Output ONE line only: a short topic label (about 4–12 words, Title Case) for the common **international / conflict / politics** theme. No JSON, no quotes, no bullets — plain text one line.

Headlines:
"""


def _merge_similar_topic_labels(
    topics_raw: list[tuple[str, str, str]],
    topic_assignments: dict[str, list[str]],
    embedding_model: str,
    *,
    threshold: float = 0.90,
) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    """Merge topics whose final labels embed too similarly (duplicate / near-duplicate)."""
    if len(topics_raw) <= 1:
        return topics_raw, topic_assignments

    labels = [t[1] for t in topics_raw]
    embs = embed_texts(labels, model_name=embedding_model)
    sim = cosine_similarity_matrix(embs)
    n = len(topics_raw)
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def article_count_for_index(ix: int) -> int:
        tid = topics_raw[ix][0]
        return sum(1 for vs in topic_assignments.values() if tid in vs)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ca, cb = article_count_for_index(ra), article_count_for_index(rb)
        if ca >= cb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    old_tid_to_canon: dict[str, str] = {}
    new_topics: list[tuple[str, str, str]] = []
    new_slugs: set[str] = set()

    for _root, idxs in groups.items():
        best_ix = max(
            idxs,
            key=lambda ix: (article_count_for_index(ix), len(topics_raw[ix][1])),
        )
        canon_tid, canon_label, _ = topics_raw[best_ix]
        for ix in idxs:
            old_tid_to_canon[topics_raw[ix][0]] = canon_tid
        slug = _slugify(canon_label)
        base = slug
        u = 0
        while slug in new_slugs:
            u += 1
            slug = f"{base}-{u}"
        new_slugs.add(slug)
        new_topics.append((canon_tid, canon_label, slug))

    new_assign: dict[str, list[str]] = {}
    for aid, tlist in topic_assignments.items():
        seen: set[str] = set()
        out: list[str] = []
        for t in tlist:
            nt = old_tid_to_canon.get(t, t)
            if nt not in seen:
                seen.add(nt)
                out.append(nt)
        new_assign[aid] = out

    if len(new_topics) < len(topics_raw):
        logger.info(
            "Topic merge: %d → %d topics (label embedding similarity ≥ %.2f)",
            len(topics_raw),
            len(new_topics),
            threshold,
        )
    return new_topics, new_assign


def _fallback_label_from_titles(titles: list[str], *, max_words: int = 10, max_chars: int = 72) -> str:
    """When the LLM returns nothing usable, derive a readable label from member headlines."""
    best = ""
    for t in titles:
        s = (t or "").strip()
        if len(s) > len(best):
            best = s
    if not best:
        return "News cluster"
    best = re.sub(r"\s+", " ", best)
    words = best.split()
    if len(words) > max_words:
        best = " ".join(words[:max_words])
    if len(best) > max_chars:
        cut = best[: max_chars - 1]
        if " " in cut:
            best = cut.rsplit(" ", 1)[0] + "…"
        else:
            best = cut + "…"
    return best[:max_chars]


def _normalize_label_llm_response(content: str) -> str:
    """Parse plain line, fenced JSON, or inline JSON for a topic label."""
    s = (content or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    blob = _extract_json_object(s)
    if blob:
        try:
            data = json.loads(blob)
            for key in ("label", "topic", "title", "topic_label", "name", "heading"):
                v = data.get(key)
                if v and str(v).strip():
                    return str(v).strip()[:160]
        except json.JSONDecodeError:
            pass
    for line in s.split("\n"):
        t = line.strip().strip('"`')
        if not t or t.lower() in ("```json", "```", "json"):
            continue
        if t.startswith("{") and "}" in t:
            continue
        if len(t) >= 3:
            return t[:160]
    return ""


def _slugify(label: str) -> str:
    s = re.sub(r"[^\w\s-]", "", label.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s or "topic"


def _label_cluster_ollama(titles: list[str], config: LLMConfig) -> str:
    """Use Ollama to produce one short topic label from a list of headlines."""
    fb = _fallback_label_from_titles(titles)
    try:
        import ollama
    except ImportError:
        logger.warning("ollama not installed; topic label from headlines")
        return fb
    sample = "\n".join(titles[:15])
    prompt = TOPIC_LABEL_PROMPT + sample
    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 256, "temperature": 0.15},
        )
        content = (response.get("message") or {}).get("content") or ""
        label = _normalize_label_llm_response(content)
        if not label:
            logger.debug("Topic label LLM empty parse; headline fallback: %s", fb[:60])
            return fb
        return label
    except Exception as e:
        logger.warning("Ollama topic label failed: %s; using headline fallback", e)
        return fb


def _label_cluster(titles: list[str], config: LLMConfig) -> str:
    if config.provider == "ollama":
        return _label_cluster_ollama(titles, config)
    return _fallback_label_from_titles(titles)


def extract_topics(
    storage: Storage,
    config: Config,
    hours_window: int = 168,  # 7 days
    max_topics: int = 25,
    min_articles_per_topic: int = 2,
    *,
    use_cluster_verdict: bool = True,
    merge_similar_topic_labels: bool = True,
    merge_topic_label_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """
    Load articles in time window, cluster by embedding similarity, optionally gate each cluster
    with a small LLM (agree / disagree / unrelated — unrelated skipped), then label and save.
    Near-duplicate topic labels (by embedding similarity) can be merged into one topic.
    Returns list of {id, label, slug, article_count}.
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
    t_emb = time.perf_counter()
    embs = embed_texts(texts, model_name=config.embedding_model)
    logger.info("Topic extraction: embedded %d articles in %.2fs", len(texts), time.perf_counter() - t_emb)
    n = len(articles)

    # Hierarchical clustering; cut to get ~max_topics clusters
    if n <= 1:
        return []
    cond = pdist(embs, metric="cosine")
    Z = linkage(cond, method="average")
    # Distance cuts: try a few thresholds. With many articles the dendrogram often jumps
    # from "too many clusters" straight to one — never hitting [2, max_topics+5], and the
    # old else branch (t=1.0) merged everything into a single topic.
    labels: np.ndarray | None = None
    for t in (0.5, 0.7, 0.9, 1.0, 1.2):
        cand = fcluster(Z, t=t, criterion="distance")
        n_c = len(set(cand))
        if 2 <= n_c <= max_topics + 5:
            labels = cand
            break
    n_clusters = len(set(labels)) if labels is not None else 0
    if labels is None or n_clusters < 2 or n_clusters > max_topics + 5:
        k = max(2, min(max_topics, n - 1))
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_clusters = len(set(labels))
        logger.info(
            "Topic clustering: using maxclust=%d → %d clusters (distance cuts did not fit [%d, %d])",
            k,
            n_clusters,
            2,
            max_topics + 5,
        )

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

    eligible_clusters = [
        (cid, members)
        for cid, members in clusters.items()
        if len(members) >= min_articles_per_topic
    ]
    n_eligible = len(eligible_clusters)
    logger.info("Topic extraction: labeling %d cluster(s) (min size %d)", n_eligible, min_articles_per_topic)

    for idx, (cid, members) in enumerate(eligible_clusters, start=1):
        t0 = time.perf_counter()
        titles = [m[1] for m in members]
        verdict_note = "n/a"
        if use_cluster_verdict and config.llm.provider == "ollama":
            tv = classify_topic_cluster_ollama(members, config.llm)
            verdict_note = tv.verdict
            if tv.verdict == "unrelated":
                logger.info(
                    "Topic %d/%d cluster_id=%s skipped (unrelated) articles=%d %.2fs",
                    idx,
                    n_eligible,
                    cid,
                    len(members),
                    time.perf_counter() - t0,
                )
                continue
            fallback = ""
            if not (tv.label or "").strip():
                fallback = _label_cluster(titles, config.llm)
            label = final_topic_label(tv, fallback)
        else:
            label = _label_cluster(titles, config.llm)
        slug = _slugify(label)
        if slug in seen_slugs:
            slug = f"{slug}-{cid}"
        seen_slugs.add(slug)
        tid = str(uuid.uuid4())
        topics_raw.append((tid, label, slug))
        for art, _ in members:
            topic_assignments.setdefault(art["id"], []).append(tid)
        logger.info(
            "Topic %d/%d done label=%r articles=%d verdict=%s cluster_id=%s %.2fs",
            idx,
            n_eligible,
            label,
            len(members),
            verdict_note,
            cid,
            time.perf_counter() - t0,
        )

    if not topics_raw:
        logger.warning("No clusters with >= %d articles after grouping", min_articles_per_topic)
        return []

    if merge_similar_topic_labels:
        topics_raw, topic_assignments = _merge_similar_topic_labels(
            topics_raw,
            topic_assignments,
            config.embedding_model,
            threshold=merge_topic_label_threshold,
        )

    logger.info("Saving %d topic(s) after clustering%s", len(topics_raw), " + verdict gate" if use_cluster_verdict else "")
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
