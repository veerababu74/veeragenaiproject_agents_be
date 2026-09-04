"""Per-project SQLite, with the retention sweep every project needs.

Each project owns one database file in the shared data directory, so projects
stay isolated while the connection handling, schema bootstrap and the
delete-after-N-hours sweep are written once here.
"""

import logging
import sqlite3

from core.config import settings

logger = logging.getLogger("veera.agents.database")


class Database:
    def __init__(self, slug: str, schema: str, filename: str = ""):
        self.slug = slug
        self.schema = schema
        self.path = settings.data_directory / (filename or f"{slug}.db")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        conn = self.connect()
        conn.executescript(self.schema)
        conn.commit()
        conn.close()
        logger.info("%s database ready at %s", self.slug, self.path)

    def purge_older_than(self, hours: int, tables: tuple[str, ...]) -> None:
        """Delete rows past the retention window. Order the tables so that
        children are removed before the rows they reference."""
        conn = self.connect()
        for table in tables:
            conn.execute(f"DELETE FROM {table} WHERE created_at < datetime('now', ?)", (f"-{hours} hours",))
        conn.commit()
        conn.close()
