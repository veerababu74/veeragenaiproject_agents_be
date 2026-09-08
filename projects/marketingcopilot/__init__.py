"""Marketing Copilot — RAG and an agent over a marketing team's own material.

Registers both hooks: `init_db` creates the schema and seeds the demo corpus at
startup, and `cleanup_expired_data` sweeps conversations, feedback and eval runs
past the platform's retention window. The seeded corpus is exempt from the sweep
because it is content, not user state.
"""

from core.registry import Project
from projects.marketingcopilot.database import cleanup_expired_data, init_db
from projects.marketingcopilot.router import router

PROJECT = Project(
    slug="marketingcopilot",
    title="Marketing Copilot",
    description=(
        "A RAG and agent system over a marketing team's documents, campaign numbers and "
        "compliance rules — with the routing, the evaluation and the monitoring shown."),
    routers=(router,),
    initialize=init_db,
    cleanup=cleanup_expired_data,
)
