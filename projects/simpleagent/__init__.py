"""SimpleAgent — one agent, up to ten tools, and a live view of how it decides."""

from core.registry import Project
from projects.simpleagent.database import cleanup_expired_data, init_db
from projects.simpleagent.routers import agent, documents, examples, runs, tools

PROJECT = Project(
    slug="simpleagent",
    title="SimpleAgent",
    description="Build one agent, attach up to ten tools, and watch every decision it makes in real time.",
    routers=(agent.router, tools.router, documents.router, runs.router, examples.router),
    initialize=init_db,
    cleanup=cleanup_expired_data,
)
