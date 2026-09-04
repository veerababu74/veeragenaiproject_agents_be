"""Pinecone access, one index per project.

Indexes are never shared between projects: an index has a fixed dimension, and
two projects embedding with different models cannot store vectors in the same
one. Within a project, each user gets their own namespace, so a search can never
reach another user's documents.
"""

from pinecone import Pinecone, ServerlessSpec

from core.config import settings

UPSERT_BATCH = 100


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(self, index_name: str, dimension: int):
        self.index_name = index_name
        self.dimension = dimension

    def _index(self):
        if not settings.pinecone_api_key:
            raise VectorStoreError("Pinecone is not configured")
        try:
            client = Pinecone(api_key=settings.pinecone_api_key)
            if not client.has_index(self.index_name):
                client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            return client.Index(self.index_name)
        except Exception as error:
            raise VectorStoreError("Could not connect to Pinecone") from error

    def upsert(self, namespace: str, document_id: str, filename: str,
               chunks: list[str], embeddings: list[list[float]]) -> None:
        vectors = [
            {
                "id": f"{document_id}:{position}",
                "values": embeddings[position],
                "metadata": {"document_id": document_id, "filename": filename,
                             "position": position, "text": chunk},
            }
            for position, chunk in enumerate(chunks)
        ]
        try:
            index = self._index()
            for start in range(0, len(vectors), UPSERT_BATCH):
                index.upsert(vectors=vectors[start:start + UPSERT_BATCH], namespace=namespace)
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError("Could not index the document chunks") from error

    def query(self, namespace: str, vector: list[float], top_k: int = 5) -> list[dict]:
        try:
            result = self._index().query(namespace=namespace, vector=vector, top_k=top_k,
                                         include_values=False, include_metadata=True)
            return [{"score": match.score, **match.metadata} for match in result.matches]
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError("Could not search the document chunks") from error

    def delete_document(self, namespace: str, document_id: str, chunk_count: int) -> None:
        if chunk_count <= 0:
            return
        try:
            self._index().delete(ids=[f"{document_id}:{i}" for i in range(chunk_count)], namespace=namespace)
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError("Could not delete the document vectors") from error
