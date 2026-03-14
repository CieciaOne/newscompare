"""Detect language and translate non-English articles to English (for topic/claim comparison)."""

from __future__ import annotations

import logging
from typing import Any

from newscompare.config import LLMConfig

logger = logging.getLogger(__name__)

TRANSLATE_PROMPT = """Translate the following text to English. Preserve factual content and numbers. Output only the translation, no commentary.
Text:
"""


def detect_language(text: str) -> str:
    """Return ISO 639-1 code (e.g. 'en', 'pl'). Uses first ~500 chars for speed."""
    if not (text or "").strip():
        return "en"
    try:
        from langdetect import detect
        sample = (text or "")[:500].strip()
        if not sample:
            return "en"
        return detect(sample)
    except Exception as e:
        logger.debug("langdetect failed: %s", e)
        return "en"


def translate_with_ollama(text: str, config: LLMConfig, max_chars: int = 4000) -> str:
    """Translate text to English using Ollama. Returns translated string or original on failure."""
    if not (text or "").strip():
        return text or ""
    try:
        import ollama
    except ImportError:
        logger.warning("ollama not installed; skipping translation")
        return text
    sample = (text or "")[:max_chars] + ("..." if len(text or "") > max_chars else "")
    prompt = TRANSLATE_PROMPT + sample
    try:
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": min(2048, config.max_tokens * 2)},
        )
        out = (response.get("message") or {}).get("content") or ""
        return out.strip() or text
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return text


def translate_article_if_needed(
    title: str,
    body: str,
    config: LLMConfig,
) -> tuple[str | None, str | None]:
    """
    If title+body are not English, translate to English. Returns (translated_title, translated_body)
    or (None, None) if already English or translation skipped.
    """
    combined = f"{title or ''}\n\n{body or ''}".strip()
    if not combined:
        return None, None
    lang = detect_language(combined)
    if lang == "en":
        return None, None
    logger.info("Translating from %s to en (title + body)", lang)
    translated_title = translate_with_ollama(title or "", config, max_chars=500)
    translated_body = translate_with_ollama(body or "", config, max_chars=3500)
    return translated_title, translated_body


def content_for_compare(article: dict[str, Any]) -> tuple[str, str]:
    """
    Return (title, body) to use for embedding, claim extraction, and comparison.
    Uses translated fields when present, otherwise original.
    """
    title = article.get("translated_title") or article.get("title") or ""
    body = article.get("translated_body") or article.get("body") or ""
    return title, body
