"""
FastAPI web application for AI News Digest (Task 3.1 & Modifications).
Strictly read-only presentation layer with server-rendered Jinja2 HTML.
"""

from pathlib import Path
from typing import Optional, Any
import urllib.parse
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.queries import (
    get_summary_stats,
    get_event_type_counts,
    get_distinct_sources,
    get_ranked_articles_for_dashboard,
    get_latest_articles,
    get_article_detail,
    score_to_stars,
    score_to_star_count,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def format_date_filter(date_str: Optional[str]) -> str:
    """Format ISO 8601 date string to clean display format like 'Sep 7, 2026 09:17'."""
    if not date_str or not isinstance(date_str, str) or not date_str.strip():
        return "Unknown date"
    raw = date_str.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%b %-d, %Y %H:%M")
    except Exception:
        return raw[:16].replace("T", " ")


def format_last_updated(date_str: Optional[str]) -> str:
    """Format last updated date header indicator like 'Sep 8, 2026 10:24 AM'."""
    if not date_str:
        return datetime.now().strftime("%b %-d, %Y %I:%M %p")
    raw = date_str.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%b %-d, %Y %I:%M %p")
    except Exception:
        return str(date_str)


def format_number_filter(val: Any) -> str:
    """Format integers with commas e.g. 2491 -> '2,491'."""
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)


def custom_url_for(name: str, **kwargs) -> str:
    """Clean URL generator supporting static assets and parameterized routes."""
    if name == "static":
        filename = kwargs.get("filename") or kwargs.get("path") or ""
        return f"/static/{str(filename).lstrip('/')}"
    elif name == "index":
        params = {k: v for k, v in kwargs.items() if v not in (None, "")}
        if params:
            return f"/?{urllib.parse.urlencode(params)}"
        return "/"
    elif name == "article_detail":
        aid = kwargs.get("article_id")
        return f"/article/{aid}"
    elif name == "about":
        return "/about"
    return "/"


def create_app() -> FastAPI:
    """Create and configure the FastAPI web application instance."""
    app = FastAPI(title="AI News Dashboard")

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Configure Jinja2 templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["format_date"] = format_date_filter
    templates.env.filters["format_last_updated"] = format_last_updated
    templates.env.filters["format_number"] = format_number_filter
    templates.env.filters["score_to_stars"] = score_to_stars
    templates.env.globals["url_for"] = custom_url_for

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        source: str = "",
        event_type: str = "",
        page: int = 1,
    ):
        clean_source = source.strip()
        clean_event_type = event_type.strip()
        safe_page = max(1, page)

        stats = get_summary_stats()
        event_types = get_event_type_counts()
        sources = get_distinct_sources()

        ranked_articles = get_ranked_articles_for_dashboard(
            limit=10,
            source=clean_source if clean_source else None,
            event_type=clean_event_type if clean_event_type else None,
        )

        latest_data = get_latest_articles(
            page=safe_page,
            per_page=20,
            source=clean_source if clean_source else None,
            event_type=clean_event_type if clean_event_type else None,
        )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_page": "home",
                "stats": stats,
                "event_types": event_types,
                "sources": sources,
                "selected_source": clean_source,
                "selected_event_type": clean_event_type,
                "ranked_articles": ranked_articles,
                "latest_articles": latest_data["articles"],
                "pagination": latest_data,
            },
        )

    @app.get("/article/{article_id}", response_class=HTMLResponse)
    async def article_detail(request: Request, article_id: int):
        article = get_article_detail(article_id)
        if not article:
            return templates.TemplateResponse(
                request=request,
                name="base.html",
                context={
                    "active_page": "article",
                    "error_title": "Article Not Found (404)",
                    "error_message": f"Article with ID {article_id} was not found.",
                },
                status_code=404,
            )

        return templates.TemplateResponse(
            request=request,
            name="article.html",
            context={
                "active_page": "article",
                "article": article,
            },
        )

    @app.get("/about", response_class=HTMLResponse)
    async def about(request: Request):
        stats = get_summary_stats()
        return templates.TemplateResponse(
            request=request,
            name="about.html",
            context={
                "active_page": "about",
                "stats": stats,
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return templates.TemplateResponse(
            request=request,
            name="base.html",
            context={
                "active_page": "",
                "error_title": "Page Not Found (404)",
                "error_message": getattr(exc, "detail", "The requested page does not exist."),
            },
            status_code=404,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    print("Starting AI News Digest Web Dashboard on http://localhost:5000 ...")
    uvicorn.run(app, host="127.0.0.1", port=5000)
