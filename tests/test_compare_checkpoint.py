"""Compare checkpoint resume helpers."""

from pathlib import Path

from newscompare.compare_checkpoint import (
    CompareCheckpoint,
    checkpoint_matches_run,
    default_checkpoint_path,
    group_signature,
    merge_completed,
    save_checkpoint,
    load_checkpoint,
)


def test_group_signature_order_independent() -> None:
    assert group_signature(["b", "a"]) == group_signature(["a", "b"])


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "cp.json"
    cp = CompareCheckpoint(
        hours=48,
        embedding_model="m1",
        claim_match_threshold=0.72,
        database=str(tmp_path / "db.sqlite"),
        completed_group_keys=["a,b"],
    )
    save_checkpoint(p, cp)
    loaded = load_checkpoint(p)
    assert loaded is not None
    assert loaded.completed_group_keys == ["a,b"]
    assert checkpoint_matches_run(
        loaded,
        hours=48,
        embedding_model="m1",
        threshold=0.72,
        database=str(tmp_path / "db.sqlite"),
    )


def test_merge_completed_appends() -> None:
    cp = CompareCheckpoint(hours=1, embedding_model="x", claim_match_threshold=0.5, database="/tmp/db")
    cp2 = merge_completed(cp, "1,2")
    cp3 = merge_completed(cp2, "3,4")
    assert cp3.completed_group_keys == ["1,2", "3,4"]


def test_default_checkpoint_beside_db() -> None:
    p = default_checkpoint_path("./data/newscompare.db")
    assert p.name == ".newscompare_compare_checkpoint.json"
    assert p.parent.name == "data"
