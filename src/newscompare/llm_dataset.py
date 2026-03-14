"""Extract claims from article text using a local LLM (Ollama)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from newscompare.claims_util import normalize_claims
from newscompare.config import LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Extract a short story summary and verifiable facts from the news text below.

First write a STORY SUMMARY: 2–4 sentences that describe what this article is about—the overall picture, main event, and outcome. This helps tell if different articles are covering the same story.

Then extract FACTS: one short sentence per verifiable fact (who did what, when, numbers, decisions). One fact per sentence. Be specific (names, dates, numbers). No meta ("The article says..."). Only facts another source could confirm or contradict.

Output a JSON object with two keys:
- "summary": string (the story summary)
- "claims": array of strings (the facts)

Example: {"summary": "On 10 March 2026 the Sejm debated X. Minister Y stated that... The vote was postponed.", "claims": ["Marszałek Czarzasty złożył wniosek o odebranie Ziobrze diety.", "Głosowanie zaplanowano na 12 marca."]}

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


def _parse_story_and_claims_from_response(raw: str) -> tuple[str, list[str]]:
    """Parse LLM output for summary + claims. Returns (summary, claims). Falls back to ("", claims) if only claims."""
    raw = raw.strip()
    summary = ""
    claims: list[str] = []
    try:
        js = _extract_json_object(raw)
        if js:
            obj = json.loads(js)
            if "summary" in obj and obj["summary"]:
                summary = str(obj["summary"]).strip()
                summary = re.sub(r"\s+", " ", summary)
            claims_raw = obj.get("claims") or obj.get("facts") or []
            if isinstance(claims_raw, list):
                claims = normalize_claims([str(c).strip() for c in claims_raw if c])
    except (json.JSONDecodeError, TypeError):
        pass
    if not claims:
        # Fallback: lines as claims
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip() and not ln.strip().startswith("{")]
        claims = normalize_claims([ln for ln in lines if len(ln) > 10][:20])
    return summary, claims


def _parse_claims_from_response(raw: str) -> list[str]:
    """Parse LLM output to list of claim strings. Cap at 20."""
    _summary, claims = _parse_story_and_claims_from_response(raw)
    return claims


def extract_claims_ollama(text: str, config: LLMConfig) -> list[str]:
    """Use Ollama API to extract claims. Requires ollama running and model pulled."""
    _summary, claims = extract_story_and_claims_ollama(text, config)
    return claims


def extract_story_and_claims_ollama(text: str, config: LLMConfig) -> tuple[str, list[str]]:
    """Use Ollama API to extract story summary and facts. Returns (summary, claims)."""
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama package not installed. Install with: poetry install --extras llm") from None

    prompt = _get_prompt(config, text, with_summary=True)
    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": min(config.max_tokens, 1536)},
        )
    except Exception as e:
        logger.warning("Ollama chat failed: %s", e)
        return "", []

    content = response.get("message", {}).get("content", "") or ""
    return _parse_story_and_claims_from_response(content)


def extract_story_and_claims(text: str, config: LLMConfig) -> tuple[str, list[str]]:
    """Extract story summary and facts. Returns (summary, claims)."""
    if config.provider == "ollama":
        return extract_story_and_claims_ollama(text, config)
    logger.warning("Unknown LLM provider %s; returning empty", config.provider)
    return "", []


def extract_claims(text: str, config: LLMConfig) -> list[str]:
    """Dispatch to configured LLM provider. Returns list of claim strings."""
    if config.provider == "ollama":
        return extract_claims_ollama(text, config)
    logger.warning("Unknown LLM provider %s; returning empty claims", config.provider)
    return []
