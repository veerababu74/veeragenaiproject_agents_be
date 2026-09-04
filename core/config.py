"""Settings shared by every project in this service.

There is one FastAPI app and one .env file at the repository root. Anything every
project needs — the platform's JWT secret, CORS origins, the Hugging Face bucket,
the Pinecone account — lives here once.

Anything that must differ *per project* (its Pinecone index, its storage prefix,
its database file) is read through `project_value`, which looks up
`<SLUG>_<KEY>` in the environment. A project therefore declares its own storage
needs in its own module and nothing here has to change when one is added.
"""

import os
from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
ENV_LOCAL_FILE = REPO_ROOT / ".env.local"


class Settings(BaseSettings):
    # Must match JWT_SECRET in veeragenai_projects_be. This service never issues
    # tokens; it only verifies the cookie that backend already set.
    jwt_secret: str
    frontend_url: str = "http://localhost:5173"
    frontend_urls: str = ""

    data_dir: str = ""
    port: int = 8004
    cleanup_interval_seconds: int = 3600
    retention_hours: int = 48

    # Shared storage accounts. Each project namespaces itself inside them: a
    # path prefix in the bucket, a dedicated index in Pinecone.
    huggingface_token: str = ""
    huggingface_bucket: str = "veera20/veeragenaiproject"
    pinecone_api_key: str = ""

    model_config = SettingsConfigDict(env_file=(ENV_FILE, ENV_LOCAL_FILE), extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_values(cls, values):
        # Hosting dashboards often define a variable with an empty value; treat that as unset.
        if isinstance(values, dict):
            return {key: value for key, value in values.items() if value != ""}
        return values

    @property
    def frontend_url_set(self) -> set[str]:
        return {
            url.strip().rstrip("/")
            for url in f"{self.frontend_url},{self.frontend_urls}".split(",")
            if url.strip()
        }

    @property
    def data_directory(self) -> Path:
        # Serverless filesystems are read-only outside the temp directory.
        directory = (Path(gettempdir()) / "veera_agents" if os.getenv("VERCEL")
                     else Path(self.data_dir) if self.data_dir else REPO_ROOT / "data")
        directory.mkdir(parents=True, exist_ok=True)
        return directory


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def project_value(slug: str, key: str, default: str = "") -> str:
    """A per-project setting, read from `<SLUG>_<KEY>`.

    For example `project_value("simpleagent", "pinecone_index")` reads
    SIMPLEAGENT_PINECONE_INDEX. Keeping the lookup dynamic means adding a
    project never requires editing this module.
    """
    return os.getenv(f"{slug.upper()}_{key.upper()}", "").strip() or default
