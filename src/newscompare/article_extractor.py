"""Extract main text from article URL."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import httpx
from goose3 import Goose

from newscompare.exceptions import ExtractionError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
USER_AGENT = "NewsCompare/0.1 (article extractor; no auth)"


def extract_body(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Fetch URL and extract main article text. On failure raises ExtractionError.
    Returns normalized plain text (no HTML).
    """
    logger.debug("Extracting body: %s", url[:70])
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPError as e:
        raise ExtractionError(f"HTTP error for {url}: {e}") from e
    except Exception as e:
        raise ExtractionError(f"Failed to fetch {url}: {e}") from e

    if not html or not html.strip():
        raise ExtractionError(f"Empty response from {url}")

    try:
        g = Goose()
        article = g.extract(raw_html=html)
        text = article.cleaned_text or ""
    except Exception as e:
        logger.warning("Goose extraction failed for %s: %s", url, e)
        raise ExtractionError(f"Extraction failed for {url}: {e}") from e

    text = _normalize(text)
    if not text.strip():
        raise ExtractionError(f"No content extracted from {url}")
    return text


def _normalize(text: str) -> str:
    """Collapse whitespace and trim."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()
