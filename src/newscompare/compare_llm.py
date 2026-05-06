"""LLM-only claim-pair agree / conflict / unrelated (Ollama)."""

from __future__ import annotations

import json
import logging
import re
import time

import numpy as np

from newscompare.config import LLMConfig

logger = logging.getLogger(__name__)

CLAIM_PAIR_PROMPT = """You are the sole judge of whether two short news claims refer to the same checkable fact.

The pairs below were pre-selected only by embedding similarity; many are false positives. Use the full claim text and outlets. Do not apply external rules of thumb—decide from meaning.

Output ONE JSON array only (no markdown, no commentary). Length must equal the number of PAIRS below.
Each element: {"p":<pair_index>,"v":"agree|conflict|unrelated"}

Definitions:
- agree: same factual proposition; no material contradiction (paraphrase, extra detail OK).
- conflict: same story axis or entities but incompatible facts (numbers, whether an event occurred, direction, attribution).
- unrelated: different facts, different events, or insufficient overlap to compare.

CLAIMS (index, outlet, text):
"""

PAIR_LINES = """
PAIRS (pair_index: claim_index_a vs claim_index_b):
"""


def pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _parse_verdict_array(raw: str) -> list[dict[str, object]]:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, object]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def _truncate(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _normalize_verdict(v_raw: object) -> str | None:
    v = str(v_raw or "").lower().strip()
    if v in ("agree", "conflict", "unrelated"):
        return v
    compact = re.sub(r"[^a-z]", "", v)
    for cand in ("agree", "conflict", "unrelated"):
        if cand in compact:
            return cand
    return None


def classify_claim_pairs_ollama(
    pairs: list[tuple[int, int]],
    article_claims: list[tuple[str, str, str]],
    config: LLMConfig,
    _sim_matrix: np.ndarray,
) -> dict[tuple[int, int], str]:
    """Batch Ollama calls; missing or failed pairs are not added (caller fills unrelated)."""
    try:
        import ollama
    except ImportError:
        logger.warning("ollama not installed; claim-pair verdicts unavailable (use unrelated)")
        return {}

    verdicts: dict[tuple[int, int], str] = {}
    batch_size = 18
    n_batches = max(1, (len(pairs) + batch_size - 1) // batch_size)
    t_all = time.perf_counter()
    logger.info(
        "Claim-pair LLM: model=%s pairs=%d batches≈%d",
        config.model,
        len(pairs),
        n_batches,
    )

    for bix, off in enumerate(range(0, len(pairs), batch_size), start=1):
        batch = pairs[off : off + batch_size]
        idx_used = sorted({i for a, b in batch for i in (a, b)})
        claim_lines = []
        for idx in idx_used:
            _aid, src, text = article_claims[idx]
            claim_lines.append(f"{idx} [{_truncate(str(src), 40)}] {_truncate(text)}")
        pair_lines = [f"{k}: {a} vs {b}" for k, (a, b) in enumerate(batch)]
        user_content = (
            CLAIM_PAIR_PROMPT
            + "\n".join(claim_lines)
            + PAIR_LINES
            + "\n".join(pair_lines)
            + "\n\nRespond with a JSON array only, length "
            + str(len(batch))
            + ", one object per pair_index 0.."
            + str(len(batch) - 1)
            + "."
        )
        t0 = time.perf_counter()
        try:
            response = ollama.chat(
                model=config.model,
                messages=[{"role": "user", "content": user_content}],
                options={"num_predict": 320, "temperature": 0.1},
            )
            raw = (response.get("message") or {}).get("content") or ""
        except Exception as e:
            logger.warning(
                "Claim-pair LLM batch %d/%d failed (%d pairs): %s",
                bix,
                n_batches,
                len(batch),
                e,
            )
            continue

        elapsed = time.perf_counter() - t0
        parsed = _parse_verdict_array(raw)
        by_p: dict[int, str] = {}
        for obj in parsed:
            p_raw = obj.get("p", obj.get("pair", obj.get("pair_index")))
            v_raw = obj.get("v", obj.get("verdict"))
            try:
                p_idx = int(p_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            v = _normalize_verdict(v_raw)
            if v is not None:
                by_p[p_idx] = v

        got = 0
        for k, (a, b) in enumerate(batch):
            v = by_p.get(k)
            if v is not None:
                verdicts[pair_key(a, b)] = v
                got += 1

        logger.info(
            "Claim-pair LLM batch %d/%d size=%d parsed=%d/%d %.2fs",
            bix,
            n_batches,
            len(batch),
            got,
            len(batch),
            elapsed,
        )

    logger.info("Claim-pair LLM done: resolved=%d/%d total %.2fs", len(verdicts), len(pairs), time.perf_counter() - t_all)
    return verdicts


def resolve_pair_verdicts(
    pairs: list[tuple[int, int]],
    article_claims: list[tuple[str, str, str]],
    sim_matrix: np.ndarray,
    llm_config: LLMConfig | None,
    *,
    heuristic_fallback: bool = False,
) -> dict[tuple[int, int], str]:
    """
    Pair verdict map. Ollama: LLM only; gaps -> unrelated unless heuristic_fallback
    fills from legacy heuristics (tests only).
    """
    verdicts: dict[tuple[int, int], str] = {}
    use_ollama = llm_config is not None and getattr(llm_config, "provider", "") == "ollama"

    if use_ollama and pairs and llm_config is not None:
        verdicts.update(classify_claim_pairs_ollama(pairs, article_claims, llm_config, sim_matrix))

    if heuristic_fallback:
        from newscompare.compare_pair_heuristic import resolve_heuristic_verdicts

        heur = resolve_heuristic_verdicts(pairs, article_claims, sim_matrix)
        for a, b in pairs:
            k = pair_key(a, b)
            if k not in verdicts:
                verdicts[k] = heur[k]
    else:
        for a, b in pairs:
            k = pair_key(a, b)
            if k not in verdicts:
                verdicts[k] = "unrelated"

    return verdicts
