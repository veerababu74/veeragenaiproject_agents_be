"""One FastAPI app hosting every agent project in this repository.

Each project in `projects/REGISTRY` contributes routers, which are mounted under
`/<slug>`, plus optional startup and retention hooks. Nothing here names a
project, so adding one never means editing this file.
"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from projects import REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("veera.agents")


async def retention_sweep():
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_seconds)
            for project in REGISTRY:
                if project.cleanup:
                    # One project's failing sweep must not stop the others.
                    try:
                        logger.info("Retention sweep for %s: %s", project.slug, project.cleanup())
                    except Exception:
                        logger.exception("Retention sweep failed for %s", project.slug)
        except asyncio.CancelledError:
            break


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting agents API | projects=%s | pinecone=%s | huggingface=%s",
                ", ".join(project.slug for project in REGISTRY) or "none",
                "configured" if settings.pinecone_api_key else "missing",
                "configured" if settings.huggingface_token else "missing")
    for project in REGISTRY:
        if project.initialize:
            project.initialize()
    sweep = asyncio.create_task(retention_sweep())
    logger.info("Startup complete | data retained for %s hours", settings.retention_hours)
    yield
    sweep.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sweep
    logger.info("Shutting down agents API")


app = FastAPI(title="Veera AI Agents API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_request(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    logger.info("%s %s -> %s | %.1f ms", request.method, request.url.path,
                response.status_code, (perf_counter() - started) * 1000)
    return response


allowed_origins = settings.frontend_url_set
if "localhost" in settings.frontend_url:
    allowed_origins.add(settings.frontend_url.replace("localhost", "127.0.0.1"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(allowed_origins),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for project in REGISTRY:
    for router in project.routers:
        app.include_router(router, prefix=project.prefix)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "veera-agents",
        "projects": [
            {"slug": project.slug, "title": project.title, "base_path": project.prefix}
            for project in REGISTRY
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True,
                reload_excludes=["*.db", "*.db-wal", "*.db-shm", "*.log", "data/*"])
