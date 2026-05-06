"""Optional legacy heuristics for claim-pair verdicts (tests / offline only)."""

from __future__ import annotations

import re

import numpy as np


def likely_same_fact_gate(claim_i: str, claim_j: str) -> bool:
    lower_i = claim_i.lower()
    lower_j = claim_j.lower()
    death_related = (
        "zginął", "zginęła", "zmarł", "zmarła", "died", "death", "śmierć", "dead",
        "killed", "casualties", "ofiary", "nie żyje",
    )
    other_event = (
        "powiązania", "links", "link", "relations", "związek", "bliskie",
        "inicjatyw", "opublikował", "wpis", "księg", "ocenił", "jako",
        "przywódc", "leader", "succeeded", "następc",
    )
    has_death_i = any(d in lower_i for d in death_related)
    has_death_j = any(d in lower_j for d in death_related)
    has_other_i = any(o in lower_i for o in other_event)
    has_other_j = any(o in lower_j for o in other_event)
    if has_death_i and has_other_j and not has_death_j:
        return False
    if has_death_j and has_other_i and not has_death_i:
        return False
    return True


def numeric_zero_conflict(claim_i: str, claim_j: str) -> bool:
    lower_i = claim_i.lower()
    lower_j = claim_j.lower()
    zero_phrases = (
        "no casualties", "no dead", "no deaths", "no one killed", "no one dead",
        "zero casualties", "zero dead", "0 dead", "0 casualties", "no injuries",
        "confirmed no", "no confirmed", "didn't kill", "no one injured",
    )

    def has_positive_number(t: str) -> bool:
        for m in re.finditer(r"\b(\d{1,3})\s*(dead|killed|casualties|injured|wounded|deadly)\b", t, re.I):
            if int(m.group(1)) > 0:
                return True
        if re.search(
            r"\b(multiple|dozens?|hundreds?|several)(\s+of)?\s+(dead|killed|casualties|injured|wounded)\b",
            t,
            re.I,
        ):
            return True
        return False

    has_zero_i = any(z in lower_i for z in zero_phrases)
    has_zero_j = any(z in lower_j for z in zero_phrases)
    has_num_i = has_positive_number(lower_i)
    has_num_j = has_positive_number(lower_j)
    if has_zero_i and has_num_j:
        return True
    if has_num_i and has_zero_j:
        return True
    return False


def _negation_or_direction_conflict(claim_i: str, claim_j: str) -> bool:
    negations = ("not", "no ", "never", "none", "neither", "n't ", "didn't", "won't", "cannot")
    lower_i = claim_i.lower()
    lower_j = claim_j.lower()
    has_neg_i = any(n in lower_i for n in negations)
    has_neg_j = any(n in lower_j for n in negations)
    if has_neg_i != has_neg_j:
        return True
    if ("increase" in lower_i or "rose" in lower_i or "grew" in lower_i) and (
        "decrease" in lower_j or "fell" in lower_j or "dropped" in lower_j
    ):
        return True
    if ("decrease" in lower_i or "fell" in lower_i or "dropped" in lower_i) and (
        "increase" in lower_j or "rose" in lower_j or "grew" in lower_j
    ):
        return True
    return False


def heuristic_pair_verdict_text(claim_i: str, claim_j: str, sim: float) -> str:
    if sim < 0.74:
        return "unrelated"
    if not likely_same_fact_gate(claim_i, claim_j):
        return "unrelated"
    if numeric_zero_conflict(claim_i, claim_j):
        return "conflict"
    if _negation_or_direction_conflict(claim_i, claim_j):
        return "conflict"
    return "agree"


def pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def resolve_heuristic_verdicts(
    pairs: list[tuple[int, int]],
    article_claims: list[tuple[str, str, str]],
    sim_matrix: np.ndarray,
) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for a, b in pairs:
        k = pair_key(a, b)
        out[k] = heuristic_pair_verdict_text(
            article_claims[a][2], article_claims[b][2], float(sim_matrix[a, b])
        )
    return out
