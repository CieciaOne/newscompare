"""CLI: fetch, compare, serve."""

from __future__ import annotations

import logging

import click
from rich.console import Console
from rich.table import Table

from newscompare.article_extractor import extract_body
from newscompare.compare import run_comparison_for_group
from newscompare.config import Config, ConfigError
from newscompare.exceptions import ExtractionError
from newscompare.feed_fetcher import fetch_all_feeds
from newscompare.grouping import group_articles
from newscompare.llm_dataset import extract_story_and_claims
from newscompare.storage import (
    Storage,
    insert_article,
    get_article_by_url,
    save_claims,
    save_story_summary,
    list_articles_needing_translation,
    update_article_translation,
)
from newscompare.topic_extraction import extract_topics
from newscompare.translation import content_for_compare, translate_article_if_needed

console = Console()


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
@click.pass_obj
def fetch(config: Config, enrich: bool) -> None:
    """Fetch all configured feeds and store articles."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    storage = Storage(config.database)
    entries = fetch_all_feeds(config.feeds, timeout=config.fetch_timeout_seconds)
    console.print(f"Fetched [green]{len(entries)}[/green] entries from {len(config.feeds)} feed(s).")

    skip_domains = {d.lower() for d in (config.skip_enrich_domains or [])}
    with storage.conn() as conn:
        for entry in entries:
            existing = get_article_by_url(conn, entry.link)
            if existing:
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
    console.print("Done. Articles stored.")


@cli.command()
@click.option("--group-by", type=click.Choice(["time", "title"]), default="time", help="Grouping strategy")
@click.option("--hours", default=24, type=int, help="Time window (hours) for grouping")
@click.option("--json", "as_json", is_flag=True, help="Output JSON")
@click.pass_obj
def compare(config: Config, group_by: str, hours: int, as_json: bool) -> None:
    """Group articles, extract claims (LLM), compare and print results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    storage = Storage(config.database)

    groups = group_articles(
        storage,
        config.embedding_model,
        hours_window=hours,
        title_similarity_threshold=config.grouping.title_similarity_threshold,
    )
    if not groups:
        console.print("[yellow]No articles to group. Run 'fetch' first.[/yellow]")
        return

    all_results: list[dict] = []
    for g in groups:
        article_ids = [a["id"] for a in g]
        # Ensure claims exist: run LLM for articles that don't have claims yet
        with storage.conn() as conn:
            for a in g:
                aid = a["id"]
                existing = conn.execute("SELECT 1 FROM claims WHERE article_id = ?", (aid,)).fetchone()
                if not existing:
                    title, body = content_for_compare(a)
                    text = (title or "") + "\n\n" + (body or "")
                    summary, claims = extract_story_and_claims(text, config.llm)
                    save_claims(conn, aid, claims)
                    save_story_summary(conn, aid, summary)

        articles, claims_meta = run_comparison_for_group(
            storage,
            article_ids,
            config.embedding_model,
            config.compare.claim_match_threshold,
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
    if as_json:
        import json
        console.print(json.dumps(all_results, indent=2))


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
@click.pass_obj
def extract_topics_cmd(config: Config, hours: int, max_topics: int) -> None:
    """Extract topics from articles (cluster + LLM labels). Run after fetch; used by UI for topic view."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    storage = Storage(config.database)
    topics = extract_topics(
        storage,
        config,
        hours_window=hours,
        max_topics=max_topics,
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
