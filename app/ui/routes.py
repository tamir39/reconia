"""UI routes. Server-rendered Jinja2 templates, progressively enhanced with HTMX.

User-facing text is Vietnamese; code and identifiers are English (POLICY.md §5).
"""

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.enums import DOC_TYPE_LABEL_VI, OUTCOME_LABEL_VI, DocType
from app.models import Document

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.globals["DOC_TYPE_LABEL_VI"] = DOC_TYPE_LABEL_VI
templates.env.globals["OUTCOME_LABEL_VI"] = OUTCOME_LABEL_VI

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@router.get("/documents", response_class=HTMLResponse)
def list_documents(
    request: Request,
    session: Session = Depends(get_session),
    doc_type: DocType | None = Query(default=None),
    q: str | None = Query(default=None),
) -> HTMLResponse:
    """Document list with filtering. The search box posts back here via HTMX."""
    stmt = select(Document).order_by(Document.doc_date.desc().nullslast(), Document.id.desc())
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    if q:
        stmt = stmt.where(Document.doc_no.ilike(f"%{q}%"))

    documents = session.execute(stmt.limit(200)).scalars().all()

    return templates.TemplateResponse(
        request,
        "documents/list.html",
        {
            "documents": documents,
            "doc_types": list(DocType),
            "selected_type": doc_type,
            "q": q or "",
        },
    )
