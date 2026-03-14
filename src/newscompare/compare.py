"""Compare claims across articles: match, agree, uncorroborated, conflict."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from newscompare.claims_util import normalize_claim
from newscompare.embeddings import embed_texts, cosine_similarity_matrix
from newscompare.storage import get_articles_with_claims
from newscompare.translation import content_for_compare

logger = logging.getLogger(__name__)


@dataclass
class ClaimWithMeta:
    article_id: str
    source_id: str
    claim_text: str
    label: str  # agreed | uncorroborated | conflict
    matched_article_ids: list[str] = field(default_factory=list)
    matched_sources: list[str] = field(default_factory=list)
    matched_claims: list[dict[str, str]] = field(default_factory=list)  # [{"source_id", "claim_text"}, ...] for agreed
    conflicting_claim: str | None = None
    conflicting_source_id: str | None = None


def _likely_same_fact(claim_i: str, claim_j: str) -> bool:
    """
    Heuristic: reject pairs that are clearly about different events (e.g. one about
    a death, the other about relations/links) to avoid false agreements from shared
    names (e.g. "Chamenei" in both but different people/events).
    """
    import re
    lower_i = claim_i.lower()
    lower_j = claim_j.lower()
    # Death / casualty event markers
    death_related = (
        "zginął", "zginęła", "zmarł", "zmarła", "died", "death", "śmierć", "dead",
        "killed", "casualties", "ofiary", "nie żyje",
    )
    # Different kind of event: relations, links, roles, initiatives (not the same fact as death)
    other_event = (
        "powiązania", "links", "link", "relations", "związek", "bliskie",
        "inicjatyw", "opublikował", "wpis", "księg", "ocenił", "jako",
        "przywódc", "leader", "succeeded", "następc",
    )
    has_death_i = any(d in lower_i for d in death_related)
    has_death_j = any(d in lower_j for d in death_related)
    has_other_i = any(o in lower_i for o in other_event)
    has_other_j = any(o in lower_j for o in other_event)
    # One claim is about death, the other about something else (links, initiative, etc.) -> different facts
    if has_death_i and has_other_j and not has_death_j:
        return False
    if has_death_j and has_other_i and not has_death_i:
        return False
    return True


def _match_claims(
    article_claims: list[tuple[str, str, str]],  # (article_id, source_id, claim_text)
    embeddings: np.ndarray,
    threshold: float,
    same_story_pairs: set[tuple[str, str]] | None = None,
) -> list[list[int]]:
    """
    For each claim index, return list of other claim indices that are matches (same fact).
    Requires similarity >= threshold and _likely_same_fact. If same_story_pairs is set, only
    match claims from articles that are in the set (same story).
    """
    n = len(article_claims)
    if n == 0:
        return []
    sim = cosine_similarity_matrix(embeddings)
    matches: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        aid_i = article_claims[i][0]
        text_i = article_claims[i][2]
        for j in range(n):
            if i == j:
                continue
            if sim[i, j] < threshold:
                continue
            if not _likely_same_fact(text_i, article_claims[j][2]):
                continue
            if same_story_pairs is not None:
                aid_j = article_claims[j][0]
                if (aid_i, aid_j) not in same_story_pairs and (aid_j, aid_i) not in same_story_pairs:
                    continue
            row.append(j)
        matches.append(row)
    return matches


def _detect_numeric_conflict(claim_i: str, claim_j: str) -> bool:
    """
    Detect fact contradictions: one says N dead/casualties/injured, other says no/zero.
    E.g. "11 dead" vs "no casualties", "confirmed 0" vs "multiple casualties".
    """
    import re
    lower_i = claim_i.lower()
    lower_j = claim_j.lower()
    # Phrases indicating zero / none
    zero_phrases = (
        "no casualties", "no dead", "no deaths", "no one killed", "no one dead",
        "zero casualties", "zero dead", "0 dead", "0 casualties", "no injuries",
        "confirmed no", "no confirmed", "didn't kill", "no one injured",
    )
    # Numbers (digits) in claim
    def has_positive_number(t: str) -> bool:
        # Skip years (4 digits) and look for counts
        for m in re.finditer(r"\b(\d{1,3})\s*(dead|killed|casualties|injured|wounded|deadly)\b", t, re.I):
            if int(m.group(1)) > 0:
                return True
        if re.search(r"\b(multiple|dozens?|hundreds?|several)\s+(dead|killed|casualties|injured)", t, re.I):
            return True
        return False

    has_zero_i = any(z in lower_i for z in zero_phrases)
    has_zero_j = any(z in lower_j for z in zero_phrases)
    has_num_i = has_positive_number(lower_i)
    has_num_j = has_positive_number(lower_j)
    if has_zero_i and has_num_j:
        return True
    if has_num_i and has_zero_j:
        return True
    return False


def _detect_conflict(claim_i: str, claim_j: str, sim: float) -> bool:
    """
    Same fact (high sim) but contradictory. Require sim >= 0.78 so we don't mark
    unrelated claims (different topics) as conflict just because they share words.
    """
    if sim < 0.74:
        return False
    if _detect_numeric_conflict(claim_i, claim_j):
        return True
    negations = ("not", "no ", "never", "none", "neither", "n't ", "didn't", "won't", "cannot")
    lower_i = claim_i.lower()
    lower_j = claim_j.lower()
    has_neg_i = any(n in lower_i for n in negations)
    has_neg_j = any(n in lower_j for n in negations)
    if has_neg_i != has_neg_j:
        return True
    if ("increase" in lower_i or "rose" in lower_i or "grew" in lower_i) and (
        "decrease" in lower_j or "fell" in lower_j or "dropped" in lower_j
    ):
        return True
    if ("decrease" in lower_i or "fell" in lower_i or "dropped" in lower_i) and (
        "increase" in lower_j or "rose" in lower_j or "grew" in lower_j
    ):
        return True
    return False


def _connected_components(n: int, other_source_matches: list[list[int]]) -> list[list[int]]:
    """Union-find: return list of components (each is list of indices)."""
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in other_source_matches[i]:
            if j > i:
                union(i, j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        comps.setdefault(root, []).append(i)
    return list(comps.values())


def compare_claims(
    article_claims: list[tuple[str, str, str]],  # (article_id, source_id, claim_text)
    embedding_model: str = "all-MiniLM-L6-v2",
    claim_match_threshold: float = 0.74,
    article_summaries: dict[str, str] | None = None,
    story_similarity_threshold: float = 0.38,
) -> list[ClaimWithMeta]:
    """
    Embed all claims, match by similarity. Agreed = same fact in >= 2 sources (clustered).
    If article_summaries is provided, only match claims from articles whose story summaries
    are similar (same story). Use a low threshold (0.38) so different phrasings of the same
    event still count as same story. Conflict = matched and contradictory.
    If most results are single-source: lower story_similarity_threshold (e.g. 0.35) or
    claim_match_threshold (e.g. 0.72), or ensure articles have story summaries.
    """
    if not article_claims:
        return []

    n_articles = len(set(c[0] for c in article_claims))
    n_claims = len(article_claims)
    logger.info(
        "Comparing %d claims from %d articles (threshold=%.2f)",
        n_claims,
        n_articles,
        claim_match_threshold,
    )

    same_story_pairs: set[tuple[str, str]] | None = None
    if article_summaries:
        aids_with_summary = [aid for aid, s in article_summaries.items() if (s or "").strip()]
        if aids_with_summary:
            summary_texts = [article_summaries[aid] for aid in aids_with_summary]
            summary_embs = embed_texts(summary_texts, model_name=embedding_model)
            summary_sim = cosine_similarity_matrix(summary_embs)
            aid_to_idx = {aid: i for i, aid in enumerate(aids_with_summary)}
            same_story_pairs = set()
            for ai in aids_with_summary:
                for aj in aids_with_summary:
                    if ai == aj:
                        same_story_pairs.add((ai, aj))
                    else:
                        idx_i, idx_j = aid_to_idx[ai], aid_to_idx[aj]
                        if summary_sim[idx_i, idx_j] >= story_similarity_threshold:
                            same_story_pairs.add((ai, aj))
            # Articles with no summary: allow matching with any other article (no story filter)
            all_aids = set(c[0] for c in article_claims)
            for aid in all_aids:
                if aid not in aids_with_summary:
                    for other in all_aids:
                        same_story_pairs.add((aid, other))
                        same_story_pairs.add((other, aid))
            if not same_story_pairs:
                same_story_pairs = None  # no constraint

    texts = [c[2] for c in article_claims]
    embs = embed_texts(texts, model_name=embedding_model)
    sim_matrix = cosine_similarity_matrix(embs)
    n = len(article_claims)
    match_indices = _match_claims(article_claims, embs, claim_match_threshold, same_story_pairs)

    if logger.isEnabledFor(logging.DEBUG) and n > 1:
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(float(sim_matrix[i, j]))
        if off_diag:
            import statistics
            logger.debug(
                "Claim similarity: min=%.3f max=%.3f mean=%.3f pairs_above_%.2f=%d",
                min(off_diag),
                max(off_diag),
                statistics.mean(off_diag),
                claim_match_threshold,
                sum(1 for s in off_diag if s >= claim_match_threshold),
            )

    other_source_matches = [
        [j for j in match_indices[i] if article_claims[j][1] != article_claims[i][1]]
        for i in range(n)
    ]
    conflict_with: list[tuple[str | None, str | None]] = [(None, None)] * n
    for i in range(n):
        _, sid, text = article_claims[i]
        for j in match_indices[i]:
            if _detect_conflict(text, article_claims[j][2], sim_matrix[i, j]):
                conflict_with[i] = (article_claims[j][2], article_claims[j][1])
                break

    components = _connected_components(n, other_source_matches)
    in_agreed_cluster: set[int] = set()
    result: list[ClaimWithMeta] = []

    for comp in components:
        sources_in_comp = set(article_claims[i][1] for i in comp)
        if len(sources_in_comp) < 2:
            continue
        if any(conflict_with[i][0] is not None for i in comp):
            continue
        in_agreed_cluster.update(comp)
        seen_source: dict[str, str] = {}
        for i in comp:
            sid = article_claims[i][1]
            text = article_claims[i][2]
            if sid not in seen_source:
                seen_source[sid] = text
        items = [(sid, text) for sid, text in seen_source.items()]
        first_aid, first_sid, first_text = article_claims[comp[0]]
        matched_claims = [{"source_id": s, "claim_text": t} for s, t in items if (s, t) != (first_sid, first_text)]
        result.append(
            ClaimWithMeta(
                article_id=first_aid,
                source_id=first_sid,
                claim_text=first_text,
                label="agreed",
                matched_article_ids=[],
                matched_sources=list(s for s, _ in items if s != first_sid),
                matched_claims=matched_claims,
                conflicting_claim=None,
                conflicting_source_id=None,
            )
        )

    for i in range(n):
        if i in in_agreed_cluster:
            continue
        aid, sid, text = article_claims[i]
        cw, cs = conflict_with[i]
        if cw is not None:
            result.append(
                ClaimWithMeta(
                    article_id=aid,
                    source_id=sid,
                    claim_text=text,
                    label="conflict",
                    matched_article_ids=[],
                    matched_sources=[],
                    matched_claims=[],
                    conflicting_claim=cw,
                    conflicting_source_id=cs,
                )
            )
        else:
            result.append(
                ClaimWithMeta(
                    article_id=aid,
                    source_id=sid,
                    claim_text=text,
                    label="uncorroborated",
                    matched_article_ids=[],
                    matched_sources=[],
                    matched_claims=[],
                    conflicting_claim=None,
                    conflicting_source_id=None,
                )
            )

    agreed = sum(1 for r in result if r.label == "agreed")
    uncorr = sum(1 for r in result if r.label == "uncorroborated")
    conflict = sum(1 for r in result if r.label == "conflict")
    logger.info("Claims: %d agreed clusters, %d uncorroborated, %d conflict", agreed, uncorr, conflict)

    return result


def run_comparison_for_group(
    storage: Any,
    article_ids: list[str],
    embedding_model: str,
    claim_match_threshold: float,
) -> tuple[list[dict[str, Any]], list[ClaimWithMeta]]:
    """
    Load claims for given article ids from storage, run compare_claims, return
    (list of article dicts, list of ClaimWithMeta).
    """
    with storage.conn() as conn:
        claims_by_article = get_articles_with_claims(conn, article_ids)
        placeholders = ",".join("?" * len(article_ids))
        rows = conn.execute(
            f"SELECT id, source_id, url, title, translated_title, translated_body, story_summary FROM articles WHERE id IN ({placeholders})",
            article_ids,
        ).fetchall()
        articles = [dict(r) for r in rows]
    for a in articles:
        a["title"] = content_for_compare(a)[0] or a.get("title") or ""
    article_summaries = {
        a["id"]: (a.get("story_summary") or "").strip()
        for a in articles
        if (a.get("story_summary") or "").strip()
    }
    # Build flat list (article_id, source_id, claim_text)
    article_claims: list[tuple[str, str, str]] = []
    for aid in article_ids:
        sid = next((a.get("source_id", "") for a in articles if a.get("id") == aid), "")
        for ct in claims_by_article.get(aid) or []:
            article_claims.append((aid, sid, normalize_claim(ct)))

    if not article_claims:
        logger.warning("No claims found for %d articles; run compare or ensure LLM extracted claims", len(article_ids))
        return articles, []

    claims_meta = compare_claims(
        article_claims,
        embedding_model=embedding_model,
        claim_match_threshold=claim_match_threshold,
        article_summaries=article_summaries or None,
    )
    return articles, claims_meta
