"""FastAPI application entrypoint."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.ui.routes import router as ui_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Reconia",
    description="Hệ thống đối soát hóa đơn và chứng từ nhập xuất",
    version="0.1.0",
    debug=get_settings().app_debug,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "ui" / "static"), name="static")
app.include_router(ui_router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe. Deliberately does not touch the database."""
    return {"status": "ok"}


@app.get("/health/db", tags=["ops"])
def health_db() -> dict[str, str]:
    """Readiness probe: confirms the database is reachable."""
    from sqlalchemy import text

    from app.db import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        return {"status": "error", "database": str(exc)}
