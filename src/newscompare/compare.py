"""Compare claims across articles: match, agree, uncorroborated, conflict."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from newscompare.claims_util import normalize_claim
from newscompare.compare_llm import pair_key, resolve_pair_verdicts
from newscompare.config import LLMConfig
from newscompare.embeddings import embed_texts, cosine_similarity_matrix
from newscompare.storage import get_articles_with_claims
from newscompare.story_schema import build_claim_embedding_text, build_same_story_embedding_text, parse_story_incident_json
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


def _match_claims(
    article_claims: list[tuple[str, str, str]],  # (article_id, source_id, claim_text)
    embeddings: np.ndarray,
    threshold: float,
    same_story_pairs: set[tuple[str, str]] | None = None,
) -> list[list[int]]:
    """
    For each claim index, return list of other claim indices with embedding similarity >= threshold.
    Optional same_story_pairs: only match claims from articles in that relation.
    """
    n = len(article_claims)
    if n == 0:
        return []
    sim = cosine_similarity_matrix(embeddings)
    matches: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        aid_i = article_claims[i][0]
        for j in range(n):
            if i == j:
                continue
            if sim[i, j] < threshold:
                continue
            if same_story_pairs is not None:
                aid_j = article_claims[j][0]
                if (aid_i, aid_j) not in same_story_pairs and (aid_j, aid_i) not in same_story_pairs:
                    continue
            row.append(j)
        matches.append(row)
    return matches


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


def _collect_candidate_pairs(match_indices: list[list[int]], n: int) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in match_indices[i]:
            if i == j:
                continue
            pk = pair_key(i, j)
            if pk not in seen:
                seen.add(pk)
                out.append(pk)
    return out


def compare_claims(
    article_claims: list[tuple[str, str, str]],  # (article_id, source_id, claim_text)
    embedding_model: str = "all-MiniLM-L6-v2",
    claim_match_threshold: float = 0.74,
    article_summaries: dict[str, str] | None = None,
    story_similarity_threshold: float = 0.38,
    article_incidents: dict[str, dict[str, str]] | None = None,
    llm_config: LLMConfig | None = None,
    *,
    claim_pair_heuristic_fallback: bool = False,
    max_claim_pairs_for_llm: int | None = None,
) -> list[ClaimWithMeta]:
    """
    Embed all claims (each claim prefixed with structured incident when available), match by
    similarity. Pairwise agree / conflict / unrelated: Ollama LLM when provider is ollama;
    gaps or non-Ollama → unrelated unless claim_pair_heuristic_fallback (tests only).

    If article_summaries is provided, only match claims from articles whose synopsis+incident
    texts are similar (same story).
    """
    if not article_claims:
        return []

    n_articles = len(set(c[0] for c in article_claims))
    n_claims = len(article_claims)
    logger.info(
        "Comparing %d claims from %d articles (threshold=%.2f, pair_llm=%s, heuristic_fallback=%s)",
        n_claims,
        n_articles,
        claim_match_threshold,
        llm_config is not None and getattr(llm_config, "provider", "") == "ollama",
        claim_pair_heuristic_fallback,
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
            all_aids = set(c[0] for c in article_claims)
            for aid in all_aids:
                if aid not in aids_with_summary:
                    for other in all_aids:
                        same_story_pairs.add((aid, other))
                        same_story_pairs.add((other, aid))
            if not same_story_pairs:
                same_story_pairs = None

    inc_map = article_incidents or {}
    texts = [
        build_claim_embedding_text(c[2], inc_map.get(c[0]))
        for c in article_claims
    ]
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

    candidate_pairs = _collect_candidate_pairs(match_indices, n)
    n_pairs_raw = len(candidate_pairs)
    if max_claim_pairs_for_llm is not None and n_pairs_raw > max_claim_pairs_for_llm:
        scored: list[tuple[float, tuple[int, int]]] = []
        for a, b in candidate_pairs:
            scored.append((float(sim_matrix[a, b]), (a, b)))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidate_pairs = [p for _, p in scored[:max_claim_pairs_for_llm]]
        logger.info(
            "Claim-pair cap: %d → %d pairs (kept highest embedding similarity; may miss rare edges)",
            n_pairs_raw,
            len(candidate_pairs),
        )

    verdicts = resolve_pair_verdicts(
        candidate_pairs,
        article_claims,
        sim_matrix,
        llm_config,
        heuristic_fallback=claim_pair_heuristic_fallback,
    )

    def vpair(i: int, j: int) -> str:
        return verdicts.get(pair_key(i, j), "unrelated")

    other_source_matches = [
        [j for j in match_indices[i] if article_claims[j][1] != article_claims[i][1] and vpair(i, j) == "agree"]
        for i in range(n)
    ]
    conflict_with: list[tuple[str | None, str | None]] = [(None, None)] * n
    for i in range(n):
        for j in match_indices[i]:
            if vpair(i, j) == "conflict":
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
    llm_config: LLMConfig | None = None,
    *,
    claim_pair_heuristic_fallback: bool = False,
    max_claim_pairs_for_llm: int | None = None,
) -> tuple[list[dict[str, Any]], list[ClaimWithMeta]]:
    """
    Load claims for given article ids from storage, run compare_claims, return
    (list of article dicts, list of ClaimWithMeta).
    """
    with storage.conn() as conn:
        claims_by_article = get_articles_with_claims(conn, article_ids)
        placeholders = ",".join("?" * len(article_ids))
        rows = conn.execute(
            f"""SELECT id, source_id, url, title, translated_title, translated_body,
                       story_summary, story_incident_json
                FROM articles WHERE id IN ({placeholders})""",
            article_ids,
        ).fetchall()
        articles = [dict(r) for r in rows]
    for a in articles:
        a["title"] = content_for_compare(a)[0] or a.get("title") or ""
    article_summaries: dict[str, str] = {}
    article_incidents: dict[str, dict[str, str]] = {}
    for a in articles:
        aid = a["id"]
        inc = parse_story_incident_json(a.get("story_incident_json"))
        article_incidents[aid] = inc
        block = build_same_story_embedding_text((a.get("story_summary") or "").strip(), inc)
        if block.strip():
            article_summaries[aid] = block
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
        article_incidents=article_incidents or None,
        llm_config=llm_config,
        claim_pair_heuristic_fallback=claim_pair_heuristic_fallback,
        max_claim_pairs_for_llm=max_claim_pairs_for_llm,
    )
    return articles, claims_meta
