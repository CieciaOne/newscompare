#!/usr/bin/env bash
# Collect → process → export for a rolling window (default 90 days).
# RSS only exposes recent items; build 2–3 month depth by running this on a schedule (e.g. daily cron).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export NEWSCOMPARE_CONFIG="${NEWSCOMPARE_CONFIG:-$ROOT/config.yaml}"

DAYS="${1:-90}"
HOURS=$((DAYS * 24))
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$ROOT/exports/$STAMP}"

echo "== fetch (insert only last ${DAYS}d by publish date when known) =="
poetry run newscompare fetch --since-days "$DAYS"

echo "== translate =="
poetry run newscompare translate --hours "$HOURS" --limit 10000

echo "== extract-topics (replaces topic assignments for this window) =="
poetry run newscompare extract-topics --hours "$HOURS" --max-topics 50

echo "== export bundle =="
poetry run newscompare export --out-dir "$OUT" --since-days "$DAYS"

echo "Done. Bundle: $OUT"
echo "Optional (LLM-heavy): poetry run newscompare compare --hours $HOURS --json -o $OUT/compare_groups.json"
