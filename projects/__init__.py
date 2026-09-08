"""The projects this service hosts.

Adding one is two steps: write it under `projects/<slug>/` exposing a `PROJECT`,
then add it to REGISTRY below. `main.py` reads only this list — it mounts every
router under `/<slug>`, runs each project's `initialize` at startup and each
project's `cleanup` on the retention schedule.
"""

from core.registry import Project
from projects.insidellm import PROJECT as INSIDELLM
from projects.marketingcopilot import PROJECT as MARKETINGCOPILOT
from projects.simpleagent import PROJECT as SIMPLEAGENT

REGISTRY: tuple[Project, ...] = (
    SIMPLEAGENT,
    INSIDELLM,
    MARKETINGCOPILOT,
)
