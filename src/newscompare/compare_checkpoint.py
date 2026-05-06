"""Resume compare runs: skip groups already completed successfully."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


@dataclass
class CompareCheckpoint:
    version: int = CHECKPOINT_VERSION
    hours: int = 0
    embedding_model: str = ""
    claim_match_threshold: float = 0.0
    database: str = ""
    completed_group_keys: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CompareCheckpoint:
        return cls(
            version=int(data.get("version", CHECKPOINT_VERSION)),
            hours=int(data.get("hours", 0)),
            embedding_model=str(data.get("embedding_model", "")),
            claim_match_threshold=float(data.get("claim_match_threshold", 0.0)),
            database=str(data.get("database", "")),
            completed_group_keys=list(data.get("completed_group_keys") or []),
        )


def group_signature(article_ids: list[str]) -> str:
    """Stable key for a compare group (order-independent)."""
    return ",".join(sorted(article_ids))


def default_checkpoint_path(database_path: str) -> Path:
    p = Path(database_path).expanduser().resolve()
    return p.parent / ".newscompare_compare_checkpoint.json"


def load_checkpoint(path: Path) -> CompareCheckpoint | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CompareCheckpoint.from_json(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Could not read compare checkpoint %s: %s", path, e)
        return None


def save_checkpoint(path: Path, cp: CompareCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cp.to_json(), indent=2), encoding="utf-8")
    tmp.replace(path)


def checkpoint_matches_run(cp: CompareCheckpoint, *, hours: int, embedding_model: str, threshold: float, database: str) -> bool:
    db_r = str(Path(database).expanduser().resolve())
    db_c = str(Path(cp.database).expanduser().resolve()) if cp.database else ""
    return (
        cp.version == CHECKPOINT_VERSION
        and cp.hours == hours
        and cp.embedding_model == embedding_model
        and abs(cp.claim_match_threshold - threshold) < 1e-9
        and db_c == db_r
    )


def merge_completed(cp: CompareCheckpoint, group_key: str) -> CompareCheckpoint:
    keys = list(cp.completed_group_keys)
    if group_key not in keys:
        keys.append(group_key)
    return CompareCheckpoint(
        version=cp.version,
        hours=cp.hours,
        embedding_model=cp.embedding_model,
        claim_match_threshold=cp.claim_match_threshold,
        database=cp.database,
        completed_group_keys=keys,
    )
