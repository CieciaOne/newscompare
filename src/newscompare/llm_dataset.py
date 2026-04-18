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

Tone: descriptive, direct, and decisive — state what happened as plainly as the text allows. Do not hedge with "may" unless the source does. If the article does not state a field, write exactly: "Not stated in the text."

Use this schema (all string values; "claims" is an array of strings):

- "synopsis": 2–4 sentences. The tight overview: who did what, why it matters, and the upshot. Same firm tone as wire copy.

- "incident": an object with these keys (each value one or two sentences unless "additional_context" needs a short third sentence):
  - "action": What occurred — the decisive event or move.
  - "driver": Stated cause, trigger, motive, or precondition (or "Not stated in the text.").
  - "outcome": Documented result, decision, toll, or declared impact.
  - "timeframe": When — absolute dates/times or relative timing exactly as given.
  - "actor": Who initiated or is responsible (person, institution, role).
  - "affected": Who or what was acted upon — target, scope, victims, beneficiaries, jurisdiction.
  - "additional_context": Location, legal basis, mechanism, figures, caveats, or other grounding the reader needs (or "None." if nothing material).

- "claims": array of atomic, verifiable facts — one short sentence each (who / what / when / numbers). No meta ("The article says…"). Facts another outlet could confirm or contradict. Maximum 20 items.

Example shape (content is illustrative):
{"synopsis":"…","incident":{"action":"…","driver":"…","outcome":"…","timeframe":"…","actor":"…","affected":"…","additional_context":"…"},"claims":["…","…"]}

Text:
"""

# Legacy prompt: claims only (no summary)
DEFAULT_CLAIMS_ONLY_PROMPT = """Extract only factual, verifiable claims from the news text below. One short sentence per claim.

Rules:
- ONE fact per claim: each sentence must state a single verifiable fact (who did what, when, numbers, decisions). Do NOT combine multiple facts in one sentence.
- Be SPECIFIC: include names, places, dates, or numbers so the claim cannot be confused with other stories.
- Do NOT include meta or background. Only state facts that another source could confirm or contradict.
- Output a JSON object with a single key "claims" containing a list of strings. No commentary.

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
        claims = normalize_claims([ln for ln in lines if len(ln) > 10][:20])
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
    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": min(max(config.max_tokens, 1536), 2048)},
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
