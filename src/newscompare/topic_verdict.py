"""Small-model LLM gate: same-event agree vs disagree vs unrelated (skip unrelated clusters)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from newscompare.config import LLMConfig
from newscompare.llm_dataset import _extract_json_object

logger = logging.getLogger(__name__)

# One JSON object; model must skip clusters outside geopolitics / conflict / world politics.
TOPIC_CLUSTER_VERDICT_PROMPT = """Classify the news lines: do they describe ONE same real-world event, and is that event **in scope** for international affairs?

Output ONE JSON object only (no markdown, no text before or after):
{"verdict":"agree","label":"Short Title Case Label","why":"<=180 chars"}

verdict must be exactly one of: agree | disagree | unrelated
- agree: same event; lines compatible (minor gaps OK); theme is war, armed conflict, security, diplomacy, sanctions, major elections or policy between states / multilateral bodies, or cross-border crisis.
- disagree: same in-scope event but clearly conflicting facts (who, numbers, whether something happened).
- unrelated: different events; OR too mixed/vague to be one story; OR the common theme is **out of scope** (e.g. sports-only, celebrity gossip, consumer tech reviews, purely local lifestyle with no international politics or conflict angle — **skip these**).

label: 4–14 words, Title Case; empty "" only if verdict is unrelated.
why: one or two short phrases (max ~180 characters); empty "" if unrelated.

Lines (outlet in brackets):
"""


@dataclass(frozen=True)
class TopicClusterVerdict:
    verdict: str  # agree | disagree | unrelated
    label: str
    why: str


def _strip_markdown_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    return s


def parse_topic_verdict_json(raw: str) -> TopicClusterVerdict:
    """Parse model output; on failure assume agree so we do not drop clusters silently."""
    cleaned = _strip_markdown_fence(raw.strip())
    blob = _extract_json_object(cleaned)
    if not blob:
        return TopicClusterVerdict("agree", "", "")
    try:
        data: dict[str, Any] = json.loads(blob)
    except json.JSONDecodeError:
        return TopicClusterVerdict("agree", "", "")
    v = str(data.get("verdict", "agree")).lower().strip()
    if v not in ("agree", "disagree", "unrelated"):
        compact = re.sub(r"[^a-z]", "", v)
        for cand in ("agree", "disagree", "unrelated"):
            if cand in compact or cand in v.replace(" ", ""):
                v = cand
                break
        else:
            v = "agree"
    label_raw = (
        data.get("label")
        or data.get("topic")
        or data.get("title")
        or data.get("topic_label")
        or data.get("name")
        or data.get("heading")
        or ""
    )
    label = str(label_raw).strip()[:240]
    why = str(data.get("why", "") or data.get("reason", "") or data.get("rationale", "")).strip()[:400]
    return TopicClusterVerdict(v, label, why)


def classify_topic_cluster_ollama(
    members: list[tuple[dict[str, Any], str]],
    config: LLMConfig,
    *,
    max_lines: int = 10,
) -> TopicClusterVerdict:
    """
    One Ollama call per cluster. members: (article row dict, display title).
    """
    try:
        import ollama
    except ImportError:
        logger.warning("ollama not installed; skipping cluster verdict gate")
        return TopicClusterVerdict("agree", "", "")

    lines: list[str] = []
    for i, (art, title) in enumerate(members[:max_lines], start=1):
        src = str(art.get("source_id") or "?")[:72]
        t = (title or "").strip()[:200]
        lines.append(f"{i}. [{src}] {t}")
    user_content = TOPIC_CLUSTER_VERDICT_PROMPT + "\n".join(lines)
    logger.info(
        "Topic cluster verdict: model=%s members=%d (sending %d lines)",
        config.model,
        len(members),
        len(lines),
    )

    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": user_content}],
            options={
                "num_predict": 640,
                "temperature": 0.1,
            },
        )
        raw = (response.get("message") or {}).get("content") or ""
    except Exception as e:
        logger.warning("Cluster verdict Ollama call failed: %s", e)
        return TopicClusterVerdict("agree", "", "")

    verdict = parse_topic_verdict_json(raw)
    if verdict.verdict == "unrelated" and verdict.why:
        logger.info("Cluster verdict unrelated: %s", verdict.why[:120])
    elif verdict.verdict == "disagree" and verdict.why:
        logger.info("Cluster verdict disagree: %s", verdict.why[:120])
    return verdict


def final_topic_label(verdict: TopicClusterVerdict, fallback_label: str) -> str:
    """Combine verdict + model label with headline-only fallback."""
    base = (verdict.label or "").strip()
    if not base:
        base = fallback_label.strip() or "News cluster"
    if verdict.verdict == "disagree":
        if not base.lower().startswith("disput"):
            return f"Disputed: {base}"
        return base
    if verdict.verdict == "unrelated":
        return base  # unused; caller skips
    return base
