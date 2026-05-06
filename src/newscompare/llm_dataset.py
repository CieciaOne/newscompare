"""Extract structured story + claims from article text using a local LLM (Ollama)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from newscompare.claims_util import normalize_claims
from newscompare.config import LLMConfig
from newscompare.story_schema import empty_incident, normalize_incident

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """You are a news desk editor. Read the article and output ONE JSON object only (no markdown fences).

Tone: descriptive, direct, decisive — state what happened as plainly as the text allows. Do not hedge with "may" unless the source does. If the article does not state a field, write exactly: "Not stated in the text."

**Scope for claims:** This project compares **international politics, war, armed conflict, security, diplomacy, sanctions, and major state or multilateral decisions**. In "claims", prioritize facts in that space (actors, military or political moves, casualties, borders, treaties, votes, sanctions, aid, energy/security policy). If the piece is **only** about sports, celebrity, consumer gadgets, or purely local human-interest with **no** geopolitical or conflict angle, return an **empty "claims" array** [] and keep synopsis brief. If the piece mixes fluff with one geopolitical fact, keep only the geopolitical claims.

Use this schema (all string values; "claims" is an array of strings):

- "synopsis": 2–4 sentences. Who did what, why it matters, upshot. Wire copy style.

- "article_sentiment": Optional one short phrase: the outlet's stance or tone toward the main subject (neutral factual, sympathetic, critical, alarmist, etc.). If not inferable, use "Not stated in the text."

- "incident": object with keys (each one or two sentences unless "additional_context" needs a short third):
  - "action": decisive event or move.
  - "driver": stated cause, trigger, motive, or precondition.
  - "outcome": documented result, toll, decision, or impact.
  - "timeframe": when, as given.
  - "actor": who initiated or is responsible.
  - "affected": target, scope, victims, beneficiaries, jurisdiction.
  - "additional_context": place, mechanism, figures, caveats; or "None." if immaterial.

- "claims": at most **12** items. Each one **substantive** checkable fact another outlet could agree or dispute (who did what, numbers, decisions). One fact per string where possible. Omit scene-setting, duplicate angles, and non-geopolitical trivia per scope above. Phrase so semantic match across outlets is possible (entities + outcomes).

Example shape (illustrative):
{"synopsis":"…","article_sentiment":"…","incident":{"action":"…","driver":"…","outcome":"…","timeframe":"…","actor":"…","affected":"…","additional_context":"…"},"claims":["…","…"]}

Text:
"""

# Legacy prompt: claims only (no summary)
DEFAULT_CLAIMS_ONLY_PROMPT = """Extract only factual, verifiable claims from the news text below. One short sentence per claim.

Rules:
- ONE fact per claim: each sentence must state a single verifiable fact (who did what, when, numbers, decisions). Do NOT combine multiple facts in one sentence.
- Be SPECIFIC: include names, places, dates, or numbers so the claim cannot be confused with other stories.
- Do NOT include meta or background. Only state facts that another source could confirm or contradict.
- Output a JSON object with a single key "claims" containing a list of strings (at most 12 substantive facts). No commentary.

Text:
"""


def _get_prompt(config: LLMConfig, text: str, max_chars: int = 8000, with_summary: bool = True) -> str:
    default = DEFAULT_PROMPT if with_summary else DEFAULT_CLAIMS_ONLY_PROMPT
    prompt_template = (config.claim_extract_prompt or default).strip()
    if not prompt_template.endswith("\n"):
        prompt_template += "\n"
    truncated = text[:max_chars] + ("..." if len(text) > max_chars else "")
    return prompt_template + truncated


def _extract_json_object(s: str) -> str | None:
    """Return first complete {...} object by brace counting."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_story_and_claims_from_response(raw: str) -> tuple[str, list[str], dict[str, str]]:
    """Parse LLM output: (synopsis, claims, incident). Legacy {"summary", "claims"} still supported."""
    raw = raw.strip()
    synopsis = ""
    claims: list[str] = []
    incident = empty_incident()
    try:
        js = _extract_json_object(raw)
        if js:
            obj = json.loads(js)
            syn_raw = obj.get("synopsis") if "synopsis" in obj else obj.get("summary")
            if syn_raw is not None:
                synopsis = str(syn_raw).strip()
                synopsis = re.sub(r"\s+", " ", synopsis)
            sent_raw = (
                obj.get("article_sentiment")
                or obj.get("stance")
                or obj.get("sentiment")
                or obj.get("editorial_tone")
            )
            if sent_raw is not None:
                sent = str(sent_raw).strip()
                if sent and sent.lower() not in ("not stated in the text.", 'not stated in the text'):
                    if synopsis:
                        synopsis = f"Editorial stance: {sent}. {synopsis}"
                    else:
                        synopsis = f"Editorial stance: {sent}."
            inc_raw = obj.get("incident")
            if isinstance(inc_raw, dict):
                incident = normalize_incident(inc_raw)
            claims_raw = obj.get("claims") or obj.get("facts") or []
            if isinstance(claims_raw, list):
                claims = normalize_claims([str(c).strip() for c in claims_raw if c])
    except (json.JSONDecodeError, TypeError):
        pass
    if not claims:
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip() and not ln.strip().startswith("{")]
        claims = normalize_claims([ln for ln in lines if len(ln) > 10][:12])
    return synopsis, claims, incident


def _parse_claims_from_response(raw: str) -> list[str]:
    """Parse LLM output to list of claim strings. Cap at 20."""
    _s, claims, _i = _parse_story_and_claims_from_response(raw)
    return claims


def extract_claims_ollama(text: str, config: LLMConfig) -> list[str]:
    """Use Ollama API to extract claims. Requires ollama running and model pulled."""
    _summary, claims, _inc = extract_story_and_claims_ollama(text, config)
    return claims


def extract_story_and_claims_ollama(text: str, config: LLMConfig) -> tuple[str, list[str], dict[str, str]]:
    """Ollama: synopsis, claims, structured incident dict."""
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama package not installed. Install with: poetry install --extras llm") from None

    prompt = _get_prompt(config, text, with_summary=True)
    # num_predict: room for full JSON without mid-sentence truncation (cap raised for larger models).
    cap = min(max(config.max_tokens, 2048), 4096)
    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": cap},
        )
    except Exception as e:
        logger.warning("Ollama chat failed: %s", e)
        return "", [], empty_incident()

    content = response.get("message", {}).get("content", "") or ""
    return _parse_story_and_claims_from_response(content)


def extract_story_and_claims(text: str, config: LLMConfig) -> tuple[str, list[str], dict[str, str]]:
    """Extract synopsis, claims, and incident. Returns ("", [], empty) if provider unsupported."""
    if config.provider == "ollama":
        return extract_story_and_claims_ollama(text, config)
    logger.warning("Unknown LLM provider %s; returning empty", config.provider)
    return "", [], empty_incident()


def extract_claims(text: str, config: LLMConfig) -> list[str]:
    """Dispatch to configured LLM provider. Returns list of claim strings."""
    if config.provider == "ollama":
        return extract_claims_ollama(text, config)
    logger.warning("Unknown LLM provider %s; returning empty claims", config.provider)
    return []
