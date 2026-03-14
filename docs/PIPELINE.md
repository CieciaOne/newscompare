# NewsCompare pipeline: from fetch to topics and comparison

This document explains how articles flow through the system and why you might see “vague” grouping or “lost” past articles.

---

## 1. Fetch (store articles)

**Command:** `newscompare fetch`

**What happens:**

- RSS/Atom feeds from `config.yaml` are fetched.
- For each entry we check **by URL** if we already have that article (`get_article_by_url`). If yes, we **skip** (no duplicate).
- New entries are **inserted** into `articles` (id, source_id, url, title, body, published_at, fetched_at). We **never delete** articles on fetch.

**Result:** All articles ever fetched stay in the DB. “Losing” them is not from fetch; it comes from how we **filter** and **reassign** in later steps.

---

## 2. Grouping (for CLI compare)

**Used by:** `newscompare compare` (and internally for grouping before claim extraction).

**What happens:**

- **Load:** All articles from DB, ordered by `fetched_at` DESC.
- **Time window:** Keep only articles whose `published_at` or `fetched_at` is within the last **`hours_window`** hours (default 24). Config: `grouping.hours_window`.
- **Embed:** Titles (using `content_for_compare`: translated title if present, else title) are embedded with the **embedding model** (e.g. `all-MiniLM-L6-v2`).
- **Cluster:** Greedy clustering by **title similarity**: two articles are in the same group if their title embedding cosine similarity ≥ **`title_similarity_threshold`** (default 0.5). Each article is in at most one group.

**Result:** List of groups. Each group is “same story” by title similarity within the time window. Articles outside the window are **not** in any group (so they don’t appear in `compare` output).

---

## 3. Topic extraction (topics in the UI)

**Command:** `newscompare extract-topics`  
**Config:** `extract-topics --hours 168` → last 7 days (default).

**What happens:**

1. **Load with time window:** Only articles with `COALESCE(published_at, fetched_at) >= cutoff` (e.g. last 168 hours). Older articles are **not** considered at all.
2. **Text for embedding:** For each article, `title + (body or translated_body)[:500]` (via `content_for_compare`). This is to cluster by “what the article is about,” not just title.
3. **Embed:** All these texts with the same **embedding model** (e.g. `all-MiniLM-L6-v2`).
4. **Hierarchical clustering:**  
   - Pairwise **cosine distance** between embeddings.  
   - **Linkage** (average).  
   - **Cut** the tree with a distance threshold `t` (tried 0.5, 0.7, 0.9, 1.0, 1.2) until number of clusters is between 2 and about `max_topics + 5` (default max 25). So we get **~25 clusters**.
5. **Filter small clusters:** Drop clusters with fewer than **`min_articles_per_topic`** (default 2).
6. **Label each cluster with LLM:** For each cluster we send a few headlines (e.g. first 15) to **Ollama** (or configured LLM) with a prompt like: “Output exactly one short topic label (2–6 words).” So each cluster gets a **label** (e.g. “Iran Strikes Israel”).
7. **Save:**  
   - **`save_topics(conn, topics_raw)`** → **deletes all rows** from `topics` and `article_topics`, then inserts the **new** topics and **new** assignments.  
   - So **every run of extract-topics fully replaces** topics and which article belongs to which topic.

**Why grouping can feel vague:**

- **Single embedding per article** (title + 500 chars): different phrasings of the same event can sit in different clusters; similar wording on different events can land in the same cluster.
- **Fixed number of clusters** (~25): the tree cut is global, so some clusters are broad (“World News”) and others narrow.
- **One topic per article:** Each article is assigned to **one** cluster only (the one it was clustered into). So an article is in exactly one topic.
- **Time window:** Only last N hours (e.g. 7 days) are clustered. Older articles are **not** in any topic after this run.

**Why “past articles” seem lost:**

- **Topics are replaced:** After each `extract-topics`, previous topic IDs and labels are gone. Only the **current** clustering exists. So “the topic I was reading” can disappear or get a new ID/label.
- **Window:** Articles older than `hours_window` are **never** in the current topic set. They still exist in `articles`; they just have **no** row in `article_topics` for the current topics.

---

## 4. Claim and story extraction (per article)

**When:** On demand when you open a topic/compare in the UI, or during `newscompare compare` for each article that doesn’t have claims yet.

**What happens:**

- We take **title + body** (using `content_for_compare`: translated if present).
- Send to **LLM** (Ollama, model from `llm.model`, e.g. `llama3.2:3b`).
- **Prompt** asks for:  
  - **Story summary:** 2–4 sentences (what the article is about).  
  - **Facts (claims):** One short sentence per verifiable fact.
- Response is parsed (JSON: `summary`, `claims`). **Normalized** (trim, collapse whitespace) and stored:
  - **`story_summary`** → `articles.story_summary`
  - **`claims`** → `claims` table (article_id, claim_text).

**Used for:** Comparison (agreed/conflict/single-source) and “same story” filtering.

---

## 5. Comparison (agreed / conflict / single-source)

**When:** You open a topic in the UI and the backend runs comparison for that topic’s articles (or you run “Compare selected” on chosen articles).

**What happens:**

1. Load **claims** for the selected article IDs.
2. **Optional “same story” filter:** If we have **story summaries** for those articles, we embed the summaries and only treat two articles as “same story” if summary similarity ≥ threshold (default 0.38). Two claims can be **agreed** only if they come from articles that are “same story” (or one has no summary).
3. **Embed** all claim texts with the same **embedding model**.
4. **Match:** Two claims are a “match” if:  
   - Cosine similarity ≥ **`claim_match_threshold`** (default 0.74), and  
   - **`_likely_same_fact`** (heuristic: e.g. not “death” vs “links”), and  
   - (If we use story filter) their articles are in **same_story_pairs**.
5. **Other source:** We only count **agreed** if the match is from a **different source**. Same source → stays single-source.
6. **Conflict:** Among matches we detect contradictions (negation, numbers, increase vs decrease) and label those **conflict**.
7. **Clusters:** Connected components of “match” form **agreed** clusters; the rest are **single-source** (or conflict if they had a contradicting match).

**Models used:**

- **Embeddings:** `embedding_model` (e.g. `all-MiniLM-L6-v2`) from config — sentence-transformers.
- **LLM:** `llm.model` (e.g. `llama3.2:3b`) — Ollama for extraction and topic labels.

---

## 6. Timeline (all articles over time)

Articles are **never** deleted on fetch. They are only **excluded** from:

- **Grouping** (CLI compare) when outside the grouping time window.
- **Topic extraction** when outside the topic extraction time window.
- **Topic list in the UI** because topic assignments are replaced each run and only cover the last N hours.

To see “how stories develop over time,” the app can show a **timeline**: all articles from the DB grouped by **day** (e.g. by `published_at` or `fetched_at`). That view is independent of topics and does not “lose” past articles — it shows the full history we have in the DB.

---

## Summary table

| Step            | Input              | Filter / transform                    | Output / storage                          |
|-----------------|--------------------|----------------------------------------|-------------------------------------------|
| Fetch           | RSS feeds          | Dedupe by URL                          | Insert new rows in `articles` (never delete) |
| Grouping        | All articles       | Time window, then title-similarity     | In-memory groups for compare              |
| Extract-topics  | Articles in window | Embed title+body, hierarchical cluster | **Replace** `topics` + `article_topics`   |
| Claim extract   | Article text       | LLM → summary + claims                 | `articles.story_summary`, `claims`        |
| Compare         | Claims + summaries | Embed, same-story, match, conflict     | Agreed / conflict / single-source         |
| Timeline        | All articles       | Group by date                          | View only (no overwrite)                  |

---

## Using a better model (e.g. Qwen 3.5)

- **LLM (extraction, topic labels):** Configured in `config.yaml` under `llm.model`. If you run **Ollama**, you can pull a Qwen model (e.g. `qwen2.5:7b` or a Qwen3 variant if available in Ollama) and set e.g. `model: qwen2.5:7b`. That can improve quality of story summaries, claims, and topic labels.  
  Qwen 3.5 collection on Hugging Face: [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35). Use the same model name in Ollama if you have it there.
- **Embeddings:** `embedding_model` is from **sentence-transformers** (e.g. `all-MiniLM-L6-v2`). For better semantic similarity you can switch to a larger or multilingual model in config (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
