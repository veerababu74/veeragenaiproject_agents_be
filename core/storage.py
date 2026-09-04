"""Original uploaded files, in the platform's shared Hugging Face bucket.

Every project gets a `Bucket` with its own path prefix, so one project's files
can never collide with another's inside the shared bucket.
"""

import os
from pathlib import Path
from tempfile import gettempdir


def _configure_cache() -> None:
    # huggingface_hub writes a cache on import; on a serverless host only the
    # temp directory is writable, so point it there before the import happens.
    if os.getenv("VERCEL"):
        cache = Path(gettempdir()) / "veera_agents" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_XET_CACHE"] = str(cache / "xet")


_configure_cache()

from huggingface_hub import HfApi

from core.config import settings


class BucketError(Exception):
    pass


class Bucket:
    def __init__(self, prefix: str):
        self.prefix = prefix.strip("/")

    def path_for(self, user_id: str, document_id: str, extension: str) -> str:
        return f"{self.prefix}/users/{user_id}/documents/{document_id}.{extension}"

    def _api(self) -> HfApi:
        if not settings.huggingface_token:
            raise BucketError("Hugging Face storage is not configured")
        return HfApi(token=settings.huggingface_token)

    def upload(self, content: bytes, remote_path: str) -> None:
        try:
            self._api().batch_bucket_files(settings.huggingface_bucket, add=[(content, remote_path)])
        except BucketError:
            raise
        except Exception as error:
            raise BucketError("Could not store the original document") from error

    def delete(self, remote_path: str) -> None:
        try:
            self._api().batch_bucket_files(settings.huggingface_bucket, delete=[remote_path])
        except BucketError:
            raise
        except Exception as error:
            raise BucketError("Could not delete the stored document") from error
