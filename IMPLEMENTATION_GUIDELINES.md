# Headline & Article Comparison Tool — Implementation Guidelines

## 1. Overview

**Goal:** A small, maintainable tool that fetches news from free feeds (RSS), compares stories across sources, and helps users see how different the coverage is and where claims agree or conflict—without paid APIs or subscriptions.

**Scope:**
- Input: RSS (or Atom) feed URLs; we use only **titles and article body text**.
- Processing: Local small LLMs (<7B) only for **comprehension and dataset creation**; heavier logic (similarity, conflict, “truth” signals) is **numeric/rule-based** to keep compute predictable.
- Output: CLI with optional simple web UI; clear presentation of **differences** and **agreement/conflict** (not absolute “true/false,” but “N sources say X”, “only A says Y”, “A says Z, B says otherwise”).

**Non-goals:** No real-time fact-checking against external DBs, no paid news APIs, no large cloud LLMs.

---

## 2. Language & Stack Choice

**Recommended: Python.**

| Concern | Python | Rust |
|--------|--------|------|
| RSS / parsing | feedparser, atoma | rss crate |
| Article extraction | goose3, readability-lxml | scraper, fewer ready libs |
| Local LLM (<7B) | llama-cpp-python, ollama, transformers | candle/llama-cpp bindings, less mature |
| Embeddings / similarity | sentence-transformers, numpy/scipy | Fewer “batteries-included” options |
| CLI / Web | click + rich, FastAPI/Flask | clap, actix-web |

Python has better coverage for “RSS → text → local LLM → embeddings/similarity” with minimal glue code. Rust is preferable only if the primary requirement were binary size or maximum throughput; for a small, maintainable tool, Python is the better fit.

---

## 3. High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Feed fetcher   │────▶│  Article extract │────▶│  Per-article store  │
│  (RSS/Atom)     │     │  (title + body)  │     │  (title, body, meta)│
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  CLI / Web UI   │◀────│  Compare &       │◀────│  Small LLM layer    │
│  (display only) │     │  score module    │     │  (claims/dataset)   │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

- **Feed fetcher:** Poll configured RSS/Atom URLs; output list of entries (title, link, published, summary if present).
- **Article extractor:** For each entry, optionally fetch `link` and get clean body text (no paywalls/subscriptions; only free content).
- **Per-article store:** In-memory or minimal DB (e.g. SQLite) holding title, body, source, url, timestamp.
- **Small LLM layer:** Runs locally; input = title + body (or truncated); output = **structured dataset** (see below), not final “true/false.”
- **Compare & score:** Pure Python + numpy: embeddings similarity, claim overlap, conflict detection. No LLM in this step.
- **CLI / Web UI:** Read-only presentation of comparison results and “who said what.”

---

## 4. Data Flow (Detailed)

1. **Config:** User provides a list of RSS feed URLs (and optionally per-feed options like max items).
2. **Fetch:** For each feed URL, fetch and parse RSS/Atom; collect entries (title, link, date, optional summary).
3. **Enrich (optional):** For each entry, HTTP GET the `link` and extract main text. If extraction fails or site requires login, keep only title + feed summary.
4. **Normalize:** One record per article: `source_id`, `url`, `title`, `body` (or summary), `published_at`.
5. **Grouping:** Group articles that refer to the “same” story (e.g. by time window + title similarity or by user-selected cluster). Only compare within a group.
6. **LLM step (per article in group):** Send title + truncated body to local small LLM. LLM returns **structured dataset**: e.g. list of short **claims/facts** (one sentence each), plus optional **entities** (names, places, numbers). No “true/false” from LLM.
7. **Numeric comparison:**  
   - Embeddings for title + body (or per-claim) with a small local model (e.g. sentence-transformers).  
   - Similarity matrix between articles (or between claims).  
   - Claim alignment: match claims across articles (e.g. by embedding similarity + threshold).  
   - **Agreement:** K sources state a similar claim → “agreed”.  
   - **Single-source:** Only one source has a claim → “uncorroborated”.  
   - **Conflict:** Two or more sources have claims that are semantically opposite or contradictory → “conflict”.
8. **Presentation:** Show per-story group: which claims are agreed, which are single-source, which conflict; and optionally highlight differing wording.

---

## 5. Component Specifications

### 5.1 Feed Fetcher

- **Libraries:** `feedparser` (RSS/Atom).
- **Input:** List of feed URLs (config file or CLI arg).
- **Output:** List of entries: `title`, `link`, `published` (datetime or None), `summary` (if present).
- **Rules:** No auth; respect `User-Agent` and cache headers to avoid hammering servers; timeout (e.g. 10s) per request. If a feed fails, log and continue with others.

### 5.2 Article Extractor

- **Purpose:** Get plain text body from `link` when the feed only has a short summary.
- **Libraries:** `goose3` or `readability-lxml` + `lxml` (Python). Prefer one and stick to it.
- **Behavior:** GET `link`; parse HTML; extract main content; strip tags and normalize whitespace. If GET fails (4xx/5xx, timeout) or extraction returns empty, store only `title` + feed `summary` as body.
- **Constraint:** Do not handle JavaScript-rendered content (no headless browser); target classic HTML articles and RSS summaries only.

### 5.3 Storage (Per-Article)

- **Schema (minimal):**
  - `id` (uuid or auto)
  - `source_id` (e.g. feed URL or short name)
  - `url`, `title`, `body`, `published_at`
  - `fetched_at`
- **Backing:** SQLite is enough; one table. No ORM required; use `sqlite3` or a single thin wrapper. Optional: in-memory only for “single run” mode.

### 5.4 Story Grouping (Same Event)

- **Goal:** Decide which articles to compare together.
- **Options (pick one or combine):**
  - **Time window:** Articles from different sources within e.g. 24–48 hours.
  - **Title similarity:** Embed title; cluster by cosine similarity above threshold.
  - **Manual:** User selects a set of articles (e.g. in Web UI) to compare.
- **Recommendation:** Time window + title embedding similarity; keep logic simple (e.g. threshold on cosine).

### 5.5 Small LLM Layer (Comprehension → Dataset)

- **Role:** Turn title + body into a **structured dataset** for later numeric use. Do **not** use LLM to decide “true/false” or to compare articles.
- **Model:** Local only, <7B parameters (e.g. 1–3B or 7B quantized). Options: **llama-cpp-python** with a small model, or **Ollama** (same constraint), or **transformers** with a small model if GPU/RAM allows.
- **Input:** Concatenate title + body; truncate to fixed token budget (e.g. 512–1024 tokens) to keep latency low.
- **Output (structured):** JSON per article, e.g.:
  - `claims`: list of strings, each one short factual claim (one sentence).
  - Optionally: `entities` (names, places, orgs), `numbers` (dates, counts), or `summary` (one paragraph). Start with `claims` only; add fields only if needed.
- **Prompt:** Instruct the model to extract factual claims only, one per line or as JSON array; no commentary, no “true/false”. Parse output (regex or JSON) and validate list length (e.g. cap at 20 claims per article).

### 5.6 Numerical Comparison & “Truth” Signals

- **Embeddings:** Use a small local model (e.g. `sentence-transformers` with something like `all-MiniLM-L6-v2` or similar). No API calls.
- **Similarity:** Cosine similarity between:
  - Option A: full article text (title + body) per article.
  - Option B: per-claim embeddings; then aggregate (e.g. average pairwise similarity between two articles’ claims).
- **Claim matching:** For each claim from each article, find claims from other articles with similarity above a threshold (e.g. 0.85). Treat as “same claim” for agreement/conflict.
- **Agreement:** If ≥2 sources have a matched “same claim” → label **agreed**.
- **Uncorroborated:** If only one source has a claim (no match in others) → label **uncorroborated**.
- **Conflict:** If two claims from different sources are matched as “same topic” but their embeddings point to opposite sentiment/meaning (e.g. “X increased” vs “X decreased”), or if we add a simple contradiction check (e.g. negation), label **conflict**. Implementation: either (a) train a tiny classifier, or (b) use similarity to a “negation” phrasing, or (c) require explicit user tagging at first; start with (b) or (c) to avoid overengineering.
- **Output:** Per story group: list of claims with labels (`agreed` / `uncorroborated` / `conflict`) and which sources said what. No scalar “truth score” required; labels + source attribution are enough.

### 5.7 How to Show “Which Is True / False”

- **Avoid:** Presenting a single “true/false” per article (we have no ground truth).
- **Do:**
  - **Per claim:** “3/5 sources agree on this.” / “Only Source A says this.” / “Source A says X; Source B says the opposite.”
  - **Per article:** List claims with badges: Agreed / Uncorroborated / In conflict.
  - **Difference view:** Side-by-side or diff-like view of **wording** (e.g. same claim, different phrasing) using aligned claims.
- **Truth interpretation:** “True” in this tool = **agreed by multiple sources**. “False” or “disputed” = **conflict** or **uncorroborated single-source**. Always show source names so the user can judge.

---

## 6. CLI Design

- **Commands (minimal):**
  - `fetch` — fetch all configured feeds and store articles (no LLM).
  - `compare [--group-by=time|title] [--hours=24]` — group articles, run LLM extraction, run comparison, print results to stdout (e.g. table or JSON).
  - `serve [--port=8080]` — start the simple web UI (optional).
- **Config:** Single config file (YAML or TOML): list of feed URLs, optional `max_articles_per_feed`, paths for DB and cache. CLI flag to pass config path.
- **Output:** Use `rich` for tables and panels so “agreed/uncorroborated/conflict” and source names are readable. Optional `--json` for piping.

---

## 7. Web UI (Optional)

- **Stack:** FastAPI (or Flask) + one of: Jinja2 + vanilla JS, or a small frontend (e.g. Vue/React single page) that calls JSON API. Prefer **simple**: server-rendered HTML + minimal JS for interactivity.
- **Screens:**
  - **Feed list:** Configured feeds and last fetch status.
  - **Article list:** List of fetched articles (filter by source, date); select a subset to “compare”.
  - **Comparison view:** For a chosen group, show:
    - Table: articles vs claims; cells show claim text and label (agreed/uncorroborated/conflict).
    - Or: list of claims with source tags and badges.
    - Link to “see wording difference” for a claim (e.g. modal with side-by-side source snippets).
- **No auth required** in the guidelines; assume local or trusted network use.

---

## 8. Project Structure (Python)

```
project_root/
  pyproject.toml or requirements.txt
  config.example.yaml
  src/
    feed_fetcher.py    # RSS fetch, returns list of entries
    article_extractor.py
    storage.py         # SQLite read/write
    grouping.py        # time + title similarity
    llm_dataset.py     # call local LLM, return claims JSON
    embeddings.py      # load model, embed text/claims
    compare.py         # similarity matrix, claim match, agree/conflict labels
    cli.py             # click commands: fetch, compare, serve
    web/
      main.py          # FastAPI app
      templates/       # if server-rendered
      static/          # if any
  tests/
    test_feed_fetcher.py
    test_compare.py
```

- One main entrypoint: `python -m src.cli` or `newscompare` after install.
- Config path: env var or `--config`; default `./config.yaml`.

---

## 9. Dependencies (Python, pinned versions)

- **Feeds:** `feedparser`
- **HTTP:** `httpx` or `requests` (timeouts, User-Agent)
- **Article extraction:** `goose3` or `readability-lxml`, `lxml`
- **Local LLM:** `llama-cpp-python` (or `ollama` client, or `transformers` + `accelerate`)
- **Embeddings:** `sentence-transformers`, `torch`
- **Numeric:** `numpy`, `scipy` (cosine, clustering if needed)
- **CLI:** `click`, `rich`
- **Config:** `pyyaml` or `toml`
- **Web (optional):** `fastapi`, `uvicorn`, `jinja2`

Keep the list minimal; no duplicate libraries for the same task.

---

## 10. Implementation Order

1. **Config + feed fetcher:** Parse config; fetch one or two RSS URLs; print entries (title, link, date).
2. **Storage:** SQLite schema; save fetched entries (title, link, summary, source, date).
3. **Article extractor:** For each stored entry with link, fetch and extract body; update DB. Handle failures gracefully.
4. **Grouping:** Implement time-window + optional title-similarity (after adding embeddings for title only).
5. **LLM dataset:** One script: read one article from DB → call local LLM → parse claims JSON → save to DB or file. Then batch for all in a group.
6. **Embeddings + compare:** Embed claims; build similarity; implement claim matching and agree/uncorroborated/conflict labels; output structured result.
7. **CLI:** Wire fetch, compare, and rich output; add `serve` that runs the web app.
8. **Web UI:** List feeds/articles; run comparison via API; render comparison view and “difference” view.

---

## 11. Style & Maintainability

- **Code style:** One style (e.g. Black + isort); line length 100 or 120. Type hints for function signatures.
- **Naming:** Clear module names (as above); functions do one thing; no global state where avoidable (pass config and DB path explicitly).
- **Errors:** Use custom exceptions (e.g. `FeedFetchError`, `ExtractionError`); log with `logging`; no silent swallows. CLI: exit codes 0/1.
- **Tests:** Unit tests for: feed parsing (with fixture XML), claim matching logic, grouping (with fixed timestamps). Mock HTTP and LLM in tests.
- **Docs:** This file is the spec; add a short README (how to install, set config, run fetch/compare, optional LLM model download). Inline comments only where logic is non-obvious.

---

## 12. What Exactly Will Be Used (Summary)

| Layer | What | How |
|-------|------|-----|
| Feeds | RSS/Atom | `feedparser`; free public feeds only |
| Article text | Title + body | Feed summary + optional GET link with `goose3`/`readability-lxml` |
| Storage | Articles + later claims | SQLite; minimal schema |
| Grouping | Same story | Time window + title embedding similarity (sentence-transformers) |
| Comprehension | Title+body → claims | Local LLM <7B via llama-cpp-python or Ollama; output JSON claims |
| Similarity / comparison | Embeddings + labels | sentence-transformers; cosine; claim match → agree/uncorroborated/conflict |
| Interface | CLI + optional Web | click + rich; FastAPI + simple HTML/JS |
| Config | Feeds, paths, limits | YAML or TOML; one file |

This gives a single, consistent implementation path: free feeds → text only → small local LLM for dataset only → numeric comparison and clear presentation of differences and agreement/conflict.
