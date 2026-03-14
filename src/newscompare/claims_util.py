"""Shared utilities for claim text: normalization and formatting."""

from __future__ import annotations

import re

# Max length for stored/compared claim text (trim with ellipsis for display only; we store full normalized)
CLAIM_MAX_LENGTH = 600


def normalize_claim(text: str | None) -> str:
    """
    Standardize claim text for storage and comparison: strip, collapse whitespace,
    single line, no leading/trailing punctuation/whitespace.
    """
    if text is None:
        return ""
    s = str(text).strip()
    # Collapse any whitespace (including newlines) to single space
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def normalize_claims(claims: list[str]) -> list[str]:
    """Normalize a list of claim strings; drop empty after normalization."""
    out: list[str] = []
    seen: set[str] = set()
    for c in claims:
        n = normalize_claim(c)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out[:20]  # cap at 20
