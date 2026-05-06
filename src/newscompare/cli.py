"""CLI: fetch, compare, serve."""

from __future__ import annotations

import json
import logging
from typing import Any
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from newscompare.article_extractor import extract_body
from newscompare.logging_util import quiet_http_loggers
from newscompare.compare import run_comparison_for_group
from newscompare.compare_checkpoint import (
    CompareCheckpoint,
    checkpoint_matches_run,
    default_checkpoint_path,
    group_signature,
    load_checkpoint,
    merge_completed,
    save_checkpoint,
)
from newscompare.config import Config, ConfigError
from newscompare.exceptions import ExtractionError
from newscompare.feed_fetcher import FeedEntry, fetch_all_feeds
from newscompare.grouping import group_articles
from newscompare.llm_dataset import extract_story_and_claims
from newscompare.storage import (
    Storage,
    insert_article,
    get_article_by_url,
    save_claims,
    save_story_summary,
    save_story_incident,
    list_articles_needing_translation,
    update_article_translation,
)
from newscompare.topic_extraction import extract_topics
from newscompare.translation import content_for_compare, translate_article_if_needed
from newscompare.export_bundle import export_for_analysis
from newscompare.gdelt_ingest import GDELT_MAX_LOOKBACK_DAYS, iter_gdelt_timerange
from newscompare.histogram_report import ascii_bars, build_report_dict, day_histogram_from_conn

console = Console()


def _parse_utc_datetime_start(value: str) -> datetime:
    s = value.strip()
    if len(s) == 10:
        d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return d.replace(hour=0, minute=0, second=0, microsecond=0)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_utc_datetime_end(value: str) -> datetime:
    s = value.strip()
    if len(s) == 10:
        d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return d.replace(hour=23, minute=59, second=59, microsecond=0)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _entry_outside_since_window(entry: FeedEntry, since_days: int | None) -> bool:
    """True if entry has parsed published time strictly before cutoff (UTC). Unknown date → keep."""
    if since_days is None or since_days <= 0:
        return False
    if entry.published is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    pub = entry.published
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return pub < cutoff


def _load_config(config_path: str | None) -> Config:
    try:
        return Config.load(config_path)
    except ConfigError as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise SystemExit(1) from e


@click.group()
@click.option("--config", "-c", type=click.Path(exists=False), default=None, help="Config file path (default: config.yaml)")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """Compare news headlines and articles across RSS feeds."""
    ctx.obj = _load_config(config)


@cli.command()
@click.option("--enrich/--no-enrich", default=True, help="Fetch full article body from link")
@click.option(
    "--since-days",
    default=None,
    type=int,
    help="Only insert items published within last N days (skip older RSS items; unknown publish date still inserted)",
)
@click.pass_obj
def fetch(config: Config, enrich: bool, since_days: int | None) -> None:
    """Fetch all configured feeds and store articles."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quiet_http_loggers()
    storage = Storage(config.database)
    entries = fetch_all_feeds(config.feeds, timeout=config.fetch_timeout_seconds)
    console.print(f"Fetched [green]{len(entries)}[/green] entries from {len(config.feeds)} feed(s).")

    skip_domains = {d.lower() for d in (config.skip_enrich_domains or [])}
    skipped_age = 0
    with storage.conn() as conn:
        for entry in entries:
            existing = get_article_by_url(conn, entry.link)
            if existing:
                continue
            if _entry_outside_since_window(entry, since_days):
                skipped_age += 1
                continue
            body = entry.summary
            if enrich and entry.link:
                skip = any(entry.link.lower().find(d) >= 0 for d in skip_domains)
                if not skip:
                    try:
                        body = extract_body(entry.link, timeout=config.fetch_timeout_seconds)
                    except ExtractionError as e:
                        logging.warning("Enrich skip %s: %s", entry.link[:60], e)
            insert_article(conn, entry, body)
    if since_days and skipped_age:
        console.print(f"Skipped [yellow]{skipped_age}[/yellow] entries older than {since_days} days (by feed date).")
    console.print("Done. Articles stored.")


@cli.command()
@click.option("--group-by", type=click.Choice(["time", "title"]), default="time", help="Grouping strategy")
@click.option("--hours", default=24, type=int, help="Time window (hours) for grouping")
@click.option("--json", "as_json", is_flag=True, help="Output JSON")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="When using --json, write to this file instead of stdout",
)
@click.option("--resume", is_flag=True, default=False, help="Skip groups already finished (see checkpoint file)")
@click.option(
    "--no-checkpoint",
    is_flag=True,
    default=False,
    help="Do not read or write compare checkpoint (disables --resume persistence)",
)
@click.option(
    "--checkpoint",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Checkpoint JSON (default: .newscompare_compare_checkpoint.json beside database)",
)
@click.option(
    "--sources",
    default=None,
    help="Comma-separated source_id values; only those articles enter grouping",
)
@click.option(
    "--max-articles",
    default=None,
    type=int,
    help="Cap articles in the time window before grouping (newest first)",
)
@click.option(
    "--max-groups",
    default=None,
    type=int,
    help="Process at most this many groups (after filters), then stop",
)
@click.option(
    "--min-sources-per-group",
    default=1,
    type=int,
    help="Skip groups with fewer distinct source_id values (use 2 for cross-outlet only)",
)
@click.option(
    "--max-claim-pairs",
    default=None,
    type=int,
    help="Per group: cap claim-pair LLM work to this many highest-similarity pairs (faster)",
)
@click.option(
    "--force-reextract",
    is_flag=True,
    default=False,
    help="Re-run LLM extraction even when claims already exist in DB",
)
@click.pass_obj
def compare(
    config: Config,
    group_by: str,
    hours: int,
    as_json: bool,
    output: str | None,
    resume: bool,
    no_checkpoint: bool,
    checkpoint: str | None,
    sources: str | None,
    max_articles: int | None,
    max_groups: int | None,
    min_sources_per_group: int,
    max_claim_pairs: int | None,
    force_reextract: bool,
) -> None:
    """Group articles, extract claims (LLM), compare and print results."""
    if output and not as_json:
        console.print("[red]--output requires --json[/red]")
        raise SystemExit(2)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quiet_http_loggers()
    storage = Storage(config.database)

    source_list = None
    if sources and sources.strip():
        source_list = [x.strip() for x in sources.split(",") if x.strip()]

    groups = group_articles(
        storage,
        config.embedding_model,
        hours_window=hours,
        title_similarity_threshold=config.grouping.title_similarity_threshold,
        source_ids=source_list,
        max_articles=max_articles,
    )
    if not groups:
        console.print("[yellow]No articles to group. Run 'fetch' first.[/yellow]")
        return

    filtered: list[list[Any]] = []
    for g in groups:
        nsrc = len({(a.get("source_id") or "") for a in g})
        if nsrc >= min_sources_per_group:
            filtered.append(g)
        else:
            logging.info(
                "Skip group: %d distinct source(s) < --min-sources-per-group=%d",
                nsrc,
                min_sources_per_group,
            )
    groups = filtered
    if not groups:
        console.print("[yellow]No groups left after --min-sources-per-group filter.[/yellow]")
        return

    db_resolved = str(Path(config.database).expanduser().resolve())
    use_checkpoint = not no_checkpoint
    cp_path = Path(checkpoint).expanduser() if checkpoint else default_checkpoint_path(config.database)
    cp = CompareCheckpoint(
        hours=hours,
        embedding_model=config.embedding_model,
        claim_match_threshold=config.compare.claim_match_threshold,
        database=db_resolved,
        completed_group_keys=[],
    )
    if use_checkpoint and resume:
        loaded = load_checkpoint(cp_path)
        if loaded and checkpoint_matches_run(
            loaded,
            hours=hours,
            embedding_model=config.embedding_model,
            threshold=config.compare.claim_match_threshold,
            database=config.database,
        ):
            cp = loaded
            logging.info("Resume: skipping %d completed group(s)", len(cp.completed_group_keys))
        elif loaded:
            logging.warning("Checkpoint ignored (different --hours, embedding model, threshold, or database path)")

    done_keys = set(cp.completed_group_keys)
    logging.info(
        "Compare: %d group(s) in last %d h (filters: sources=%s max_articles=%s)",
        len(groups),
        hours,
        source_list or "all",
        max_articles or "none",
    )
    all_results: list[dict] = []
    ran = 0
    for gi, g in enumerate(groups, start=1):
        if max_groups is not None and ran >= max_groups:
            logging.info("Stopped after --max-groups=%d", max_groups)
            break
        article_ids = [a["id"] for a in g]
        gkey = group_signature(article_ids)
        if use_checkpoint and resume and gkey in done_keys:
            logging.info("Resume skip group %d/%d (%d articles)", gi, len(groups), len(article_ids))
            continue

        titles = [((content_for_compare(a)[0] or a.get("title") or "")[:60]) for a in g[:2]]
        logging.info(
            "Compare group %d/%d: %d articles; sample titles: %s",
            gi,
            len(groups),
            len(article_ids),
            titles,
        )
        with storage.conn() as conn:
            for a in g:
                aid = a["id"]
                existing = conn.execute("SELECT 1 FROM claims WHERE article_id = ?", (aid,)).fetchone()
                if not existing or force_reextract:
                    if existing and force_reextract:
                        conn.execute("DELETE FROM claims WHERE article_id = ?", (aid,))
                    title, body = content_for_compare(a)
                    text = (title or "") + "\n\n" + (body or "")
                    summary, claims, incident = extract_story_and_claims(text, config.llm)
                    save_claims(conn, aid, claims)
                    save_story_summary(conn, aid, summary)
                    save_story_incident(conn, aid, json.dumps(incident, ensure_ascii=False))

        articles, claims_meta = run_comparison_for_group(
            storage,
            article_ids,
            config.embedding_model,
            config.compare.claim_match_threshold,
            llm_config=config.llm,
            max_claim_pairs_for_llm=max_claim_pairs,
        )
        if as_json:
            all_results.append({
                "articles": [{"id": x["id"], "source_id": x["source_id"], "title": x["title"]} for x in articles],
                "claims": [
                    {
                        "article_id": c.article_id,
                        "source_id": c.source_id,
                        "claim_text": c.claim_text,
                        "label": c.label,
                        "matched_article_ids": c.matched_article_ids,
                        "conflicting_claim": c.conflicting_claim,
                    }
                    for c in claims_meta
                ],
            })
        else:
            _print_group_rich(articles, claims_meta)

        if use_checkpoint:
            cp = merge_completed(cp, gkey)
            save_checkpoint(cp_path, cp)
            done_keys.add(gkey)
        ran += 1

    if as_json:
        payload = json.dumps(all_results, indent=2)
        if output:
            Path(output).write_text(payload, encoding="utf-8")
            console.print(f"Wrote [green]{output}[/green] ({len(all_results)} group(s)).")
        else:
            console.print(payload)


@cli.command("export")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, writable=True),
    default=None,
    help="Output directory (default: exports/UTC-timestamp under cwd)",
)
@click.option(
    "--since-days",
    default=None,
    type=int,
    help="Only export articles (and their claims) with published/fetched time in last N days",
)
@click.pass_obj
def export_cmd(config: Config, out_dir: str | None, since_days: int | None) -> None:
    """Export stats.json, articles.jsonl, claims.csv, topics.json for offline analysis."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base = Path(out_dir) if out_dir else Path("exports") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    storage = Storage(config.database)
    with storage.conn() as conn:
        stats = export_for_analysis(conn, base, since_days=since_days)
    console.print(f"Export ready: [green]{base.resolve()}[/green]")
    console.print(f"  articles={stats['article_count']} claims={stats['claim_count']} topics={stats['topic_count']}")


@cli.command("db-histogram")
@click.option(
    "--source-like",
    default=None,
    help="Filter source_id with SQL LIKE, e.g. GDELT:%% or %%BBC%%",
)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write full report (articles_by_day, gap_days, …) to this JSON file",
)
@click.option("--days", default=200, type=int, help="Max lines in ASCII chart (newest days first)")
@click.pass_obj
def db_histogram(config: Config, source_like: str | None, json_out: str | None, days: int) -> None:
    """Show per-day article counts (histogram) and calendar gaps between first and last day (read-only)."""
    storage = Storage(config.database)
    with storage.conn() as conn:
        by_day, used = day_histogram_from_conn(conn, source_like=source_like)
    report = build_report_dict(by_day, used_rows=used)
    console.print(
        f"Dated articles: [cyan]{report['total_articles_with_date']}[/cyan] "
        f"across [cyan]{report['distinct_days_with_articles']}[/cyan] days "
        f"([dim]{report['first_day']} … {report['last_day']}[/dim], span {report['calendar_span_days_inclusive']}d)"
    )
    console.print(
        f"Calendar gaps (zero articles in range): [yellow]{report['gap_days_count']}[/yellow] days"
        + (" (list truncated in JSON if huge)" if report.get("gap_days_truncated") else "")
    )
    if report["gap_days_count"] and report["gap_days_count"] <= 40:
        console.print("[dim]" + ", ".join(report["gap_days"]) + "[/dim]")
    elif report["gap_days_count"]:
        console.print(f"[dim]First gaps: {', '.join(report['gap_days'][:15])} …[/dim]")
    # ASCII chart: subset by days
    if days > 0:
        top_days = sorted(by_day.items(), key=lambda x: x[0], reverse=True)[:days]
        console.print(ascii_bars(Counter(dict(top_days)), max_lines=days))
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"Wrote [green]{json_out}[/green]")


@cli.command("ingest-gdelt")
@click.option("--query", required=True, help='GDELT query, e.g. ukraine or "(climate OR warming)"')
@click.option(
    "--start",
    "start_s",
    required=True,
    help="Start (UTC): YYYY-MM-DD or ISO datetime",
)
@click.option(
    "--end",
    "end_s",
    required=True,
    help="End (UTC): YYYY-MM-DD or ISO datetime",
)
@click.option(
    "--chunk-hours",
    default=48,
    type=int,
    help="Time slice per API call (smaller if you hit maxrecords=250 often)",
)
@click.option(
    "--sleep",
    default=2.5,
    type=float,
    help="Seconds between successful chunk requests (raise if you see HTTP 429)",
)
@click.option(
    "--max-articles",
    default=None,
    type=int,
    help="Max articles to insert this run (after merge); default unlimited",
)
@click.option("--enrich/--no-enrich", default=False, help="Fetch full HTML body (slow); default off")
@click.option(
    "--since-days",
    default=None,
    type=int,
    help="Same as fetch: skip insert if published older than N days",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Less GDELT progress logging (chunk lines)",
)
@click.pass_obj
def ingest_gdelt(
    config: Config,
    query: str,
    start_s: str,
    end_s: str,
    chunk_hours: int,
    sleep: float,
    max_articles: int | None,
    enrich: bool,
    since_days: int | None,
    quiet: bool,
) -> None:
    """Backfill from GDELT DOC 2.0 (free). Only the last ~90 days are searchable; range is clipped."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quiet_http_loggers()
    if not quiet:
        logging.getLogger("newscompare.gdelt_ingest").setLevel(logging.INFO)
    start = _parse_utc_datetime_start(start_s)
    end = _parse_utc_datetime_end(end_s)
    console.print(
        f"GDELT policy: only about the last [cyan]{GDELT_MAX_LOOKBACK_DAYS}[/cyan] days from “now”; "
        "each request returns at most 250 hits."
    )
    entries, warnings = iter_gdelt_timerange(
        query,
        start,
        end,
        chunk_hours=chunk_hours,
        sleep_seconds=sleep,
        timeout=float(config.fetch_timeout_seconds) + 35.0,
        verbose=not quiet,
    )
    for w in warnings:
        logging.warning("%s", w)
    if max_articles is not None and max_articles > 0:
        entries = entries[:max_articles]
    console.print(f"GDELT returned [green]{len(entries)}[/green] unique URLs after merge.")

    skip_domains = {d.lower() for d in (config.skip_enrich_domains or [])}
    storage = Storage(config.database)
    inserted = 0
    skipped = 0
    total = len(entries)
    log_every = max(250, total // 20) if total > 500 else max(50, total // 5) if total else 1
    with storage.conn() as conn:
        for i, entry in enumerate(entries, start=1):
            existing = get_article_by_url(conn, entry.link)
            if existing:
                skipped += 1
                continue
            if _entry_outside_since_window(entry, since_days):
                skipped += 1
                continue
            body = entry.summary
            if enrich and entry.link:
                dom_skip = any(entry.link.lower().find(d) >= 0 for d in skip_domains)
                if not dom_skip:
                    try:
                        body = extract_body(entry.link, timeout=config.fetch_timeout_seconds)
                    except ExtractionError as e:
                        logging.warning("Enrich skip %s: %s", entry.link[:60], e)
            insert_article(conn, entry, body)
            inserted += 1
            if not quiet and total and (i % log_every == 0 or i == total):
                logging.info(
                    "GDELT DB %d/%d: inserted=%d skipped_dup_or_age=%d",
                    i,
                    total,
                    inserted,
                    skipped,
                )
    console.print(f"Done. Inserted [green]{inserted}[/green], skipped (dup/age) [yellow]{skipped}[/yellow].")


def _print_group_rich(articles: list[dict], claims_meta: list) -> None:
    table = Table(title="Claims by source")
    table.add_column("Source", style="cyan")
    table.add_column("Claim", style="white")
    table.add_column("Label", style="green")
    table.add_column("Conflict / Notes", style="yellow")
    for c in claims_meta:
        label_style = {"agreed": "green", "uncorroborated": "yellow", "conflict": "red"}.get(c.label, "white")
        conflict_note = c.conflicting_claim[:80] + "..." if c.conflicting_claim and len(c.conflicting_claim) > 80 else (c.conflicting_claim or "")
        if c.label == "agreed":
            conflict_note = f"Matched with {len(c.matched_article_ids)} other(s)"
        table.add_row(c.source_id, c.claim_text[:100] + ("..." if len(c.claim_text) > 100 else ""), f"[{label_style}]{c.label}[/{label_style}]", conflict_note)
    console.print(table)
    console.print()


@cli.command()
@click.option("--hours", default=168, type=int, help="Time window (hours) for articles to cluster")
@click.option("--max-topics", default=25, type=int, help="Max number of topics")
@click.option(
    "--no-cluster-verdict",
    is_flag=True,
    default=False,
    help="Disable small LLM gate (agree/disagree/unrelated); use headline-only labels as before",
)
@click.option(
    "--no-merge-similar-labels",
    is_flag=True,
    default=False,
    help="Do not merge topics whose final labels embed very similarly (default: merge)",
)
@click.option(
    "--merge-label-threshold",
    default=0.90,
    type=float,
    help="Cosine similarity (0–1) to merge topic labels after extraction (default 0.90)",
)
@click.pass_obj
def extract_topics_cmd(
    config: Config,
    hours: int,
    max_topics: int,
    no_cluster_verdict: bool,
    no_merge_similar_labels: bool,
    merge_label_threshold: float,
) -> None:
    """Extract topics from articles (cluster + LLM labels). Run after fetch; used by UI for topic view."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quiet_http_loggers()
    storage = Storage(config.database)
    topics = extract_topics(
        storage,
        config,
        hours_window=hours,
        max_topics=max_topics,
        use_cluster_verdict=not no_cluster_verdict,
        merge_similar_topic_labels=not no_merge_similar_labels,
        merge_topic_label_threshold=merge_label_threshold,
    )
    if not topics:
        console.print("[yellow]No topics extracted. Run 'fetch' first and ensure enough articles.[/yellow]")
        return
    console.print(f"Extracted [green]{len(topics)}[/green] topics.")
    for t in topics:
        console.print(f"  • {t['label']} ({t['article_count']} articles)")


@cli.command()
@click.option("--hours", default=168, type=int, help="Only articles fetched in last N hours")
@click.option("--limit", default=200, type=int, help="Max articles to process")
@click.pass_obj
def translate(config: Config, hours: int, limit: int) -> None:
    """Detect language and translate non-English articles to English (for topics/claims)."""
    from datetime import datetime, timedelta, timezone

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quiet_http_loggers()
    storage = Storage(config.database)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    with storage.conn() as conn:
        to_process = list_articles_needing_translation(conn, since=since, limit=limit)
    if not to_process:
        console.print("[green]No articles needing translation.[/green]")
        return
    console.print(f"Processing [cyan]{len(to_process)}[/cyan] articles for translation...")
    done = 0
    for a in to_process:
        title = a.get("title") or ""
        body = a.get("body") or ""
        trans_title, trans_body = translate_article_if_needed(title, body, config.llm)
        # If already English we store original so we don't re-run; otherwise store translation
        save_title = trans_title if trans_title is not None else title
        save_body = trans_body if trans_body is not None else body
        with storage.conn() as conn:
            update_article_translation(conn, a["id"], save_title, save_body)
        done += 1
    console.print(f"Done. Updated [green]{done}[/green] articles.")


@cli.command()
@click.option("--port", default=8080, type=int, help="Port for web UI")
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level (INFO shows comparison progress)")
@click.pass_obj
def serve(config: Config, port: int, host: str, log_level: str) -> None:
    """Start the web UI server."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(levelname)s [%(name)s] %(message)s",
    )
    quiet_http_loggers()
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # quiet HTTP access log by default
    try:
        from newscompare.web.main import create_app
    except ImportError as e:
        console.print("[red]Web extras not installed. Run: poetry install --extras web[/red]")
        raise SystemExit(1) from e
    app = create_app(config)
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    cli(obj=None)


if __name__ == "__main__":
    main()
