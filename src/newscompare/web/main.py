"""FastAPI app for web UI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from newscompare.compare import run_comparison_for_group
from newscompare.config import Config
from newscompare.grouping import group_articles
from newscompare.llm_dataset import extract_story_and_claims
from newscompare.translation import content_for_compare
from newscompare.storage import (
    Storage,
    save_claims,
    save_story_summary,
    save_story_incident,
    list_topics,
    get_topic,
    get_article_by_id,
    get_claims_for_article,
    get_articles_by_topic,
    get_articles_with_claims,
    list_articles_by_day,
    save_topic_compare_report,
    get_topic_compare_report,
    list_reports,
)

_web_dir = Path(__file__).resolve().parent
_templates_dir = _web_dir / "templates"
_static_dir = _web_dir / "static"


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="NewsCompare")
    app.state.config = config
    app.state.storage = Storage(config.database)

    if _templates_dir.exists():
        templates = Jinja2Templates(directory=str(_templates_dir))
    else:
        templates = None

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if templates is None:
            return HTMLResponse("<h1>NewsCompare</h1><p>No templates dir. Configure web/templates.</p>")
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            rows = conn.execute(
                "SELECT id, source_id, url, title, fetched_at, translated_title, translated_body FROM articles ORDER BY fetched_at DESC LIMIT 100"
            ).fetchall()
        articles = []
        for r in rows:
            d = dict(r)
            d["title"] = content_for_compare(d)[0] or d.get("title") or ""
            articles.append(d)
        feeds = [{"name": f.source_id, "url": f.url} for f in config.feeds]
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "articles": articles, "feeds": feeds},
        )

    @app.get("/api/articles")
    async def api_articles(request: Request):
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            rows = conn.execute(
                "SELECT id, source_id, url, title, fetched_at, translated_title, translated_body FROM articles ORDER BY fetched_at DESC LIMIT 500"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["title"] = content_for_compare(d)[0] or d.get("title") or ""
            out.append(d)
        return {"articles": out}

    @app.get("/api/articles/timeline")
    async def api_articles_timeline(request: Request, days: int = 30):
        """Articles grouped by day (published_at or fetched_at), for timeline view. Does not drop old articles."""
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            days_list = list_articles_by_day(conn, days=days)
        out = []
        for date_str, arts in days_list:
            for a in arts:
                a["title"] = content_for_compare(a)[0] or a.get("title") or ""
            out.append({"date": date_str, "articles": arts})
        return {"days": out}

    @app.get("/api/articles/{article_id}")
    async def api_article_detail(request: Request, article_id: str):
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            article = get_article_by_id(conn, article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            claims = get_claims_for_article(conn, article_id)
        title, body = content_for_compare(article)
        si_raw = article.get("story_incident_json") or ""
        try:
            story_incident = json.loads(si_raw) if si_raw.strip() else {}
        except json.JSONDecodeError:
            story_incident = {}
        return {
            "id": article["id"],
            "source_id": article.get("source_id"),
            "url": article.get("url"),
            "title": title or article.get("title"),
            "body": body or article.get("body"),
            "story_summary": article.get("story_summary") or "",
            "story_incident": story_incident if isinstance(story_incident, dict) else {},
            "published_at": article.get("published_at"),
            "fetched_at": article.get("fetched_at"),
            "claims": claims,
        }

    @app.post("/api/compare")
    async def api_compare(request: Request):
        body = await request.json()
        article_ids = body.get("article_ids") or []
        if len(article_ids) < 2:
            return {"error": "Select at least 2 articles to compare."}
        config: Config = request.app.state.config
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            for aid in article_ids:
                row = conn.execute(
                    "SELECT id, title, body, translated_title, translated_body FROM articles WHERE id = ?", (aid,)
                ).fetchone()
                if not row:
                    continue
                existing = conn.execute("SELECT 1 FROM claims WHERE article_id = ?", (aid,)).fetchone()
                if not existing:
                    d = dict(zip(["id", "title", "body", "translated_title", "translated_body"], row))
                    title, body = content_for_compare(d)
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
        )
        def claim_to_dict(c):
            return {
                "article_id": c.article_id,
                "source_id": c.source_id,
                "claim_text": c.claim_text,
                "label": c.label,
                "matched_article_ids": c.matched_article_ids,
                "matched_sources": getattr(c, "matched_sources", []),
                "matched_claims": getattr(c, "matched_claims", []),
                "conflicting_claim": c.conflicting_claim,
                "conflicting_source_id": getattr(c, "conflicting_source_id", None),
            }
        return {
            "articles": [{"id": a["id"], "source_id": a["source_id"], "title": a["title"], "url": a.get("url")} for a in articles],
            "claims": [claim_to_dict(c) for c in claims_meta],
        }

    @app.get("/api/groups")
    async def api_groups(request: Request):
        config: Config = request.app.state.config
        storage: Storage = request.app.state.storage
        groups = group_articles(
            storage,
            config.embedding_model,
            hours_window=config.grouping.hours_window,
            title_similarity_threshold=config.grouping.title_similarity_threshold,
        )
        return {
            "groups": [
                [{"id": a["id"], "source_id": a["source_id"], "title": a["title"]} for a in g]
                for g in groups
            ],
        }

    @app.get("/api/topics")
    async def api_topics_list(request: Request):
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            topics = list_topics(conn)
        return {"topics": topics}

    @app.get("/api/reports")
    async def api_reports_list(request: Request, topic_id: str | None = None, limit: int = 50):
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            reports = list_reports(conn, topic_id=topic_id, limit=limit)
        return {"reports": reports}

    @app.get("/api/topics/{topic_id}")
    async def api_topic_detail(request: Request, topic_id: str):
        storage: Storage = request.app.state.storage
        with storage.conn() as conn:
            topic = get_topic(conn, topic_id)
            if not topic:
                raise HTTPException(status_code=404, detail="Topic not found")
            articles = get_articles_by_topic(conn, topic_id, order_by_time=True)
        for a in articles:
            a["title"] = content_for_compare(a)[0] or a.get("title") or ""
        # Group by source for side-by-side; add time bucket (day) for timeline
        by_source: dict[str, list[dict]] = {}
        for a in articles:
            src = a.get("source_id") or "Unknown"
            by_source.setdefault(src, [])
            ts = a.get("published_at") or a.get("fetched_at") or ""
            day = ts[:10] if ts else ""
            by_source[src].append({**a, "day": day})
        return {
            "topic": topic,
            "articles": articles,
            "by_source": by_source,
            "sources": list(by_source.keys()),
        }

    @app.get("/api/topics/{topic_id}/compare")
    async def api_topic_compare(
        request: Request,
        topic_id: str,
        sources: str | None = None,
        labels: str | None = None,
        regenerate: bool = False,
    ):
        storage: Storage = request.app.state.storage
        config: Config = request.app.state.config
        sources_key = ",".join(sorted(s.strip() for s in (sources or "").split(",") if s.strip())) if sources else ""

        with storage.conn() as conn:
            topic = get_topic(conn, topic_id)
            if not topic:
                raise HTTPException(status_code=404, detail="Topic not found")
            articles = get_articles_by_topic(conn, topic_id, order_by_time=True)
        if sources:
            want = {s.strip() for s in sources.split(",") if s.strip()}
            articles = [a for a in articles if (a.get("source_id") or "") in want]
        if not articles:
            return {"topic": topic, "articles": [], "claims": [], "by_source": {}, "sources": [], "from_cache": False}

        if not regenerate:
            with storage.conn() as conn:
                cached = get_topic_compare_report(conn, topic_id, sources_key)
            if cached:
                logger.info("Comparison: loaded from cache (%d articles)", len(articles))
                claims_out, _ = cached
                for a in articles:
                    a["title"] = content_for_compare(a)[0] or a.get("title") or ""
                by_source = {}
                for a in articles:
                    src = a.get("source_id") or "Unknown"
                    by_source.setdefault(src, [])
                    ts = a.get("published_at") or a.get("fetched_at") or ""
                    day = ts[:10] if ts else ""
                    by_source[src].append({**a, "day": day})
                arts_list = [{"id": a["id"], "source_id": a.get("source_id"), "url": a.get("url"), "title": a["title"]} for a in articles]
                if labels:
                    want_labels = {s.strip().lower() for s in labels.split(",") if s.strip()}
                    claims_out = [c for c in claims_out if c["label"].lower() in want_labels]
                return {
                    "topic": topic,
                    "articles": arts_list,
                    "claims": claims_out,
                    "by_source": by_source,
                    "sources": list(by_source.keys()),
                    "from_cache": True,
                }

        article_ids = [a["id"] for a in articles]
        logger.info("Comparison: %d articles, computing (not from cache)", len(articles))
        with storage.conn() as conn:
            need_claims = []
            for a in articles:
                existing = conn.execute("SELECT 1 FROM claims WHERE article_id = ?", (a["id"],)).fetchone()
                if not existing:
                    need_claims.append(a["id"])
            if need_claims:
                n = len(need_claims)
                est_min = max(1, (n * 15) // 60)  # ~15s per article for LLM
                logger.info("Extracting claims for %d articles (LLM), est. ~%d min", n, est_min)
            for i, aid in enumerate(need_claims):
                row = conn.execute(
                    "SELECT id, title, body, translated_title, translated_body FROM articles WHERE id = ?", (aid,)
                ).fetchone()
                if not row:
                    continue
                logger.info("  Story+claims extraction %d/%d", i + 1, len(need_claims))
                d = dict(zip(["id", "title", "body", "translated_title", "translated_body"], row))
                title, body = content_for_compare(d)
                text = (title or "") + "\n\n" + (body or "")
                summary, claims, incident = extract_story_and_claims(text, config.llm)
                save_claims(conn, aid, claims)
                save_story_summary(conn, aid, summary)
                save_story_incident(conn, aid, json.dumps(incident, ensure_ascii=False))
        logger.info("Running claim comparison (embedding + matching)...")
        arts_list, claims_meta = run_comparison_for_group(
            storage,
            article_ids,
            config.embedding_model,
            config.compare.claim_match_threshold,
        )
        for a in articles:
            a["title"] = content_for_compare(a)[0] or a.get("title") or ""
        by_source = {}
        for a in articles:
            src = a.get("source_id") or "Unknown"
            by_source.setdefault(src, [])
            ts = a.get("published_at") or a.get("fetched_at") or ""
            day = ts[:10] if ts else ""
            by_source[src].append({**a, "day": day})

        def claim_to_dict(c):
            return {
                "article_id": c.article_id,
                "source_id": c.source_id,
                "claim_text": c.claim_text,
                "label": c.label,
                "matched_article_ids": c.matched_article_ids,
                "matched_sources": getattr(c, "matched_sources", []),
                "matched_claims": getattr(c, "matched_claims", []),
                "conflicting_claim": c.conflicting_claim,
                "conflicting_source_id": getattr(c, "conflicting_source_id", None),
            }
        claims_out = [claim_to_dict(c) for c in claims_meta]

        arts_serializable = [{"id": a["id"], "source_id": a.get("source_id"), "url": a.get("url"), "title": a.get("title")} for a in arts_list]
        with storage.conn() as conn:
            save_topic_compare_report(conn, topic_id, sources_key, claims_out, arts_serializable)
        logger.info("Comparison done: %d claims, report saved", len(claims_out))

        if labels:
            want_labels = {s.strip().lower() for s in labels.split(",") if s.strip()}
            claims_out = [c for c in claims_out if c["label"].lower() in want_labels]
        return {
            "topic": topic,
            "articles": arts_list,
            "claims": claims_out,
            "by_source": by_source,
            "sources": list(by_source.keys()),
            "from_cache": False,
        }

    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    return app
