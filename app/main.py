import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from app.config import settings
from app.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("legal_hil")

app = FastAPI(
    title="Ethiopian Legal AI HIL Pipeline",
    description="Professional AI pipeline for extracting, aligning, and human-in-the-loop verifying bilingual Ethiopian legal texts.",
    version="1.0.0"
)

# Enable CORS for local dev (Vite, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Mount frontend directory for static assets and index.html
frontend_dir = settings.BASE_DIR / "app" / "frontend"

if frontend_dir.exists():
    dist_dir = frontend_dir / "dist"
    if (dist_dir / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

    static_assets = frontend_dir / "static"
    if static_assets.exists():
        app.mount("/static", StaticFiles(directory=str(static_assets)), name="static")

    @app.get("/")
    def serve_frontend():
        dist_index = frontend_dir / "dist" / "index.html"
        if dist_index.exists():
            return FileResponse(str(dist_index))
        return RedirectResponse(url="http://localhost:5173")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
