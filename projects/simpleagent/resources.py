"""This project's slice of the shared storage accounts.

Everything project-specific about storage is declared here and nowhere else: the
Pinecone index it owns and the prefix it writes under in the shared bucket. Both
can be overridden with SIMPLEAGENT_PINECONE_INDEX and SIMPLEAGENT_STORAGE_PREFIX.
"""

from core.config import project_value
from core.embeddings import EMBEDDING_DIMENSION
from core.storage import Bucket
from core.vectors import VectorStore

SLUG = "simpleagent"

bucket = Bucket(project_value(SLUG, "storage_prefix", SLUG))
vectors = VectorStore(project_value(SLUG, "pinecone_index", "simpleagent-rag"), EMBEDDING_DIMENSION)
