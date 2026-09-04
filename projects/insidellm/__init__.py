"""Inside an LLM — a visual walkthrough of a real transformer forward pass.

Read-only: there is no database, no user state and nothing to expire, so this
project registers neither an initialize nor a cleanup hook. It is the simplest
possible thing the registry can host.
"""

from core.registry import Project
from projects.insidellm.router import router

PROJECT = Project(
    slug="insidellm",
    title="Inside an LLM",
    description="Watch a real GPT-2 forward pass, component by component, on ten fixed examples.",
    routers=(router,),
)
