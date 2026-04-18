# Plan: 2–3 month window, process, export for downstream analysis

## Reality check (RSS)

Feeds only return the **latest N items** (often tens to low hundreds). You **cannot** download a full 2–3 month archive in one `fetch` from RSS alone. **Strategy:** run `fetch` regularly (daily/weekly). The DB **dedupes by URL** and **never deletes** articles on fetch, so the database accumulates depth over calendar time.

Use **`fetch --since-days 90`** if you want to skip inserting very old items that sometimes appear in feeds (optional hygiene).

## GDELT backfill (free, bulk now — with API limits)

The [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) is free, no key, returns article URLs/titles/snippets for a **keyword query** over a time range. **Hard limits:**

- **Rolling window:** `STARTDATETIME` / `ENDDATETIME` must fall inside roughly the **last 90 days** from GDELT’s “now” (policy described by GDELT; the CLI clips your range).
- **Max ~250 articles per HTTP request;** if a slice is “full,” lower **`--chunk-hours`** (e.g. 12 or 6) and/or narrow the query.
- **Be polite:** default **`--sleep 1`** between requests.

Example (UTC dates):

```bash
poetry run newscompare ingest-gdelt \
  --query '"artificial intelligence"' \
  --start 2026-01-20 --end 2026-04-15 \
  --chunk-hours 24 \
  --max-articles 3000
```

Inserts into the **same SQLite DB** as RSS (`source_id` like `GDELT:bbc.co.uk`). Dedupe by URL; **`--no-enrich`** is default (snippet only; much faster than fetching every HTML page).

## One-command pipeline

From repo root (Poetry env active):

```bash
chmod +x scripts/pipeline_analysis.sh
./scripts/pipeline_analysis.sh 90    # days window; default 90
```

Or manually:

1. **Collect:** `poetry run newscompare fetch --since-days 90`
2. **Process:**  
   - `poetry run newscompare translate --hours 2160 --limit 10000` (90×24)  
   - `poetry run newscompare extract-topics --hours 2160 --max-topics 50`
3. **Export:** `poetry run newscompare export --since-days 90 --out-dir exports/my_run`

## Export layout

Each run writes:

| File | Purpose |
|------|---------|
| `stats.json` | Counts, date range, articles per source, per day (last 120 days in table) |
| `articles.jsonl` | One JSON object per line: id, source, url, title, times, body/story sizes, claim_count |
| `claims.csv` | claim_id, article_id, source_id, claim_text |
| `topics.json` | Current topics + article_count (last `extract-topics` run defines assignments) |

Load in R/Pandas/Excel for your larger analysis project.

## Optional: structured compare output

Claim-level agree/conflict labels are produced per **group** (same-story clustering). Heavy on LLM if the window is huge. Example:

```bash
poetry run newscompare compare --hours 168 --json -o exports/compare_week.json
```

Tune `--hours` to a smaller window for experiments.

## Speed / optimization

- **Ingest:** RSS `fetch --no-enrich`; GDELT default is already no full-page fetch. Use **`--sleep`** only as low as you are comfortable with for GDELT.
- **Translate:** Raise **`--limit`** only when needed; shrink **`--hours`** while iterating.
- **Topics:** Lower **`--max-topics`** for faster clustering/LLM labeling; widen **`--hours`** only for final runs.
- **Compare / claims:** Shorter **`--hours`** during dev; one **`--json -o`** dump when satisfied.
- **Ollama:** set **`OLLAMA_NUM_PARALLEL=1`** (default) to avoid RAM spikes; close other heavy apps.
- **Embeddings:** first run downloads the sentence-transformers model; reuse same `embedding_model` in config.

## Models (MacBook Air M3, 16 GB unified memory)

Rough guidance for **local Ollama** (GGUF quant sizes vary by build):

| Tier | Examples (Ollama names may vary) | Use |
|------|-----------------------------------|-----|
| Fast / low RAM | `llama3.2:3b`, `qwen2.5:3b`, `phi3:mini` | Quick claim extraction, labels |
| Balanced | `qwen2.5:7b`, `llama3.1:8b`, `mistral:7b` (often Q4) | Better JSON/claims; usually fits next to embeddings + OS if one model loaded at a time |
| Heavier | `gemma2:9b`, `qwen2.5:14b` | Higher quality; may be tight — watch Activity Monitor, use smaller context |

**Embeddings:** `all-MiniLM-L6-v2` is the default (fast). For more languages (e.g. Polish + English), try **`paraphrase-multilingual-MiniLM-L12-v2`** in `config.yaml` — larger download, better cross-language similarity.

**Apple Silicon:** [Ollama](https://ollama.com) uses Metal on M3. Alternatives such as **MLX**-based runners exist in the ecosystem for speed experiments; NewsCompare is wired to Ollama’s HTTP API today.

Pull then point `config.yaml` → `llm.model` at the tag you installed, e.g. `ollama pull qwen2.5:7b` then `model: qwen2.5:7b`.
