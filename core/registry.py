"""What a project is, as far as the shared app is concerned.

A project hands over its routers and two optional callables. `main.py` mounts
every router under `/<slug>` and runs every `initialize` at startup and every
`cleanup` on the retention schedule, so adding a project means writing the
project and registering it — never editing the app.
"""

from dataclasses import dataclass, field
from typing import Callable

from fastapi import APIRouter


@dataclass(frozen=True)
class Project:
    slug: str
    title: str
    description: str
    routers: tuple[APIRouter, ...] = field(default_factory=tuple)
    # Create tables and indexes. Runs once at startup.
    initialize: Callable[[], None] | None = None
    # Delete data past the retention window. Runs hourly; returns a short
    # summary the sweep logs.
    cleanup: Callable[[], object] | None = None

    @property
    def prefix(self) -> str:
        return f"/{self.slug}"
