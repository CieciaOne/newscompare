"""Canonical structured story (incident) for extraction and context-aware comparison."""

from __future__ import annotations

import json
from typing import Any

# JSON / DB keys (stable). Prompt maps human labels to these.
INCIDENT_KEYS: tuple[str, ...] = (
    "action",  # what happened — one firm sentence
    "driver",  # stated cause, trigger, or motive (or explicit "Not stated")
    "outcome",  # documented result or impact
    "timeframe",  # when, as given (absolute or relative)
    "actor",  # who initiated / primary agent
    "affected",  # target, scope, or who/what was acted upon
    "additional_context",  # place, mechanism, figures, legal basis, caveats (short)
)

# LLM / legacy aliases → canonical key
INCIDENT_ALIASES: dict[str, str] = {
    "cause": "driver",
    "precipitant": "driver",
    "result": "outcome",
    "when": "timeframe",
    "who_did_it": "actor",
    "primary_actor": "actor",
    "to_whom": "affected",
    "recipient": "affected",
    "extra_information": "additional_context",
    "context": "additional_context",
}


def empty_incident() -> dict[str, str]:
    return {k: "" for k in INCIDENT_KEYS}


def normalize_incident(raw: Any) -> dict[str, str]:
    """Merge arbitrary dict into canonical incident fields (non-empty strings)."""
    out = empty_incident()
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        if not isinstance(key, str):
            continue
        nk = key.strip().lower().replace(" ", "_").replace("-", "_")
        k = INCIDENT_ALIASES.get(nk, nk)
        if k not in out:
            continue
        s = str(val).strip() if val is not None else ""
        if s:
            out[k] = s
    return out


def incident_narrative_block(incident: dict[str, str] | None, *, max_chars: int = 2000) -> str:
    """Single text block for embeddings: firm labels + values."""
    inc = incident or empty_incident()
    lines: list[str] = []
    labels = {
        "action": "Action",
        "driver": "Driver or stated cause",
        "outcome": "Outcome",
        "timeframe": "Timeframe",
        "actor": "Actor",
        "affected": "Affected party or scope",
        "additional_context": "Additional context",
    }
    for k in INCIDENT_KEYS:
        v = (inc.get(k) or "").strip()
        if not v:
            continue
        lines.append(f"{labels[k]}: {v}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def parse_story_incident_json(raw: str | None) -> dict[str, str]:
    if not (raw or "").strip():
        return empty_incident()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return empty_incident()
    if not isinstance(obj, dict):
        return empty_incident()
    return normalize_incident(obj)


def build_same_story_embedding_text(synopsis: str, incident: dict[str, str] | None) -> str:
    """Text used to decide if articles cover the same story (embeddings)."""
    parts: list[str] = []
    s = (synopsis or "").strip()
    if s:
        parts.append("Synopsis:\n" + s)
    block = incident_narrative_block(incident)
    if block:
        parts.append("Structured incident:\n" + block)
    return "\n\n".join(parts).strip()


def build_claim_embedding_text(claim: str, incident: dict[str, str] | None, *, max_chars: int = 3500) -> str:
    """Prefix claim with incident so similarity reflects situation, not wording only."""
    claim = (claim or "").strip()
    block = incident_narrative_block(incident, max_chars=max_chars - 200)
    if not block:
        return claim[:max_chars]
    out = (
        "Use the following situation as grounding. Then judge the atomic claim.\n\n"
        f"{block}\n\n---\nAtomic claim:\n{claim}"
    )
    if len(out) > max_chars:
        return out[: max_chars - 3] + "..."
    return out
