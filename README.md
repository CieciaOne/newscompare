# NewsCompare

Compare news headlines and articles across RSS feeds: see how coverage differs and which claims are agreed, uncorroborated, or in conflict. No paid APIs or subscriptions; local LLMs only (optional, for claim extraction).

## Setup

**Poetry (recommended)**

```bash
poetry install
# Optional: claim extraction via Ollama
poetry install --extras llm
# Optional: web UI
poetry install --extras web
# All extras
poetry install --extras llm --extras web
```

**Or venv + pip**

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
# For claim extraction (Ollama): pip install ollama
# For web UI: pip install fastapi uvicorn jinja2
```

**Config**

```bash
cp config.example.yaml config.yaml
# Edit config.yaml: add your RSS feed URLs and optional LLM/embedding settings.
# For better extraction/labels, use a stronger Ollama model (e.g. qwen2.5:7b).
```

See **[docs/PIPELINE.md](docs/PIPELINE.md)** for how fetch, grouping, topic extraction, and comparison work (and why past articles can seem “lost” until you use the Timeline view).

## Usage

- **Fetch** articles from configured feeds (and optionally enrich with full body from article URL):
  ```bash
  poetry run newscompare fetch
  # or: python -m newscompare.cli fetch
  ```

- **Compare** (group articles, extract claims via local LLM if installed, then compare):
  ```bash
  poetry run newscompare compare
  poetry run newscompare compare --hours 48 --json
  ```

- **Extract topics** (batch: cluster articles + LLM labels; used by the web UI for topic-based view):
  ```bash
  poetry run newscompare extract-topics --hours 168 --max-topics 25
  ```
  Run after `fetch`. Topics are stored in the DB and shared across all sources so you can compare coverage by topic.

- **Web UI** (if extras installed):
  ```bash
  poetry run newscompare serve --port 8080
  ```
  Open http://127.0.0.1:8080. Progress and logs for comparison (claim extraction, matching) are printed in the **terminal** where `serve` runs. Use `--log-level INFO` (default) to see them; use `--log-level DEBUG` for more detail (e.g. `poetry run newscompare serve --log-level DEBUG`).

  **Topics** tab: auto-generated topics (run `extract-topics` first); click a topic to see a **timeline** (articles by day), **side-by-side by source**, and **claims** (agreed/uncorroborated/conflict). **All articles** tab: select articles and compare manually.

## Local LLM (claim extraction)

For claim extraction the app can use **Ollama** (default in config). Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2:3b
```

Set in `config.yaml` under `llm`:

- `provider: ollama`
- `model: llama3.2:3b`

Run `newscompare compare`; the first run will download the embedding model (sentence-transformers) and may take a moment.

**Comparison:** Claims are matched by embedding similarity. **Agreed** = the same fact is stated by at least one *other source* (same outlet doesn’t count). **Conflict** = similar claim but contradictory (e.g. “11 dead” vs “no casualties”, or negation). Default `claim_match_threshold` is **0.60**; raise it (e.g. 0.72) if you get false agreed, lower it if cross-source paraphrases still show uncorroborated. Set `LOG_LEVEL=DEBUG` to log similarity stats.

## Fetch logs explained

When you run `newscompare fetch`:

- **RSS feeds** — Fetched first. If you see `Fetched N entries from M feed(s)` with N &gt; 0, feeds are OK.
- **Enrich (full article body)** — For each entry we optionally GET the article URL and extract main text (goose3).
  - **"No content extracted"** — The page is video, paywalled, or JS-heavy; goose3 found no main text. We still store **title + RSS summary**.
  - **"HTTP error … 403 Forbidden"** — The site blocks scrapers (e.g. NYT, some others). We still store **title + RSS summary** from the feed; you get headlines and short descriptions for comparison, just not full body.
  - **"Publish date … could not be resolved to UTC"** — The feed had a relative date (e.g. "7 hours ago"); we keep the article but without a precise time.
- **Result** — Every article is stored. When enrichment fails, body is the feed summary only. Comparison and claim extraction still work on title + whatever body we have.

**What to do:** No action required. In `config.yaml` you can set **`skip_enrich_domains`** (e.g. `nytimes.com`, `wyborcza.pl`) so we never fetch full article body for those hosts — you get titles and feed summaries only, and no 403 / "No content extracted" spam. Use `--no-enrich` to skip enrichment for all feeds.

## Project layout

- `src/newscompare/` — package
  - `config.py` — YAML config
  - `feed_fetcher.py` — RSS/Atom fetch
  - `article_extractor.py` — full-text extraction (goose3)
  - `storage.py` — SQLite
  - `grouping.py` — time + title similarity grouping
  - `llm_dataset.py` — Ollama claim extraction
  - `embeddings.py` — sentence-transformers
  - `compare.py` — claim matching and agree/uncorroborated/conflict
  - `cli.py` — fetch / compare / serve
  - `web/` — FastAPI app and templates
- `tests/` — pytest
- `IMPLEMENTATION_GUIDELINES.md` — full spec

## Tests

```bash
poetry run pytest
# or with venv: pytest
```

## License

Use as you like.
