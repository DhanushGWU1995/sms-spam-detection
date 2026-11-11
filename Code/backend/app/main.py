from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import ROOT_DIR, get_settings
from backend.app.core.s3_store import get_store

settings = get_settings()
app = FastAPI(
    title=settings.project_name,
    description="Open-source agentic RAG consumer fraud assistant (EC2 + S3 only). Data files are fetched from S3.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes must be registered before the Angular static mount.
app.include_router(router, prefix="/api")

# Viz PNGs are durable in S3; mount the ephemeral cache directory.
VIZ_DIR = settings.cache_dir / settings.viz_key
try:
    store = get_store()
    for rel in store.list_keys(settings.viz_key.rstrip("/") + "/", suffix=".png"):
        store.download_file(rel)
except Exception:
    # Startup should not fail if AWS creds are temporarily unavailable.
    pass
VIZ_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/viz/files", StaticFiles(directory=VIZ_DIR), name="viz-files")

ANGULAR_DIST_CANDIDATES = [
    ROOT_DIR / "frontend" / "dist" / "scamshield-ui" / "browser",
    ROOT_DIR / "frontend" / "dist" / "scamshield-ui",
]
ANGULAR_DIST = next((path for path in ANGULAR_DIST_CANDIDATES if (path / "index.html").exists()), None)


@app.get("/api")
def api_root():
    return {
        "project": settings.project_name,
        "docs": "/docs",
        "s3_uri": settings.s3_uri,
        "data_storage": "s3",
        "cache_dir": str(settings.cache_dir),
    }


if ANGULAR_DIST is not None:
    # Serves index.html plus hashed JS/CSS assets for the Angular TypeScript UI.
    app.mount("/", StaticFiles(directory=ANGULAR_DIST, html=True), name="angular")
else:

    @app.get("/")
    def index():
        return {
            "project": settings.project_name,
            "message": "Angular TypeScript UI not built yet. From frontend/: npm run build",
            "api_docs": "/docs",
            "s3_uri": settings.s3_uri,
        }
