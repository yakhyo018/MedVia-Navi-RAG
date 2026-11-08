import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from navi.core.config import Settings
from navi.domain import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def point_id(doc_id: str, chunk_index: int) -> str:
    """
    Stable id from (doc_id, chunk index): the same chunk of the same document always lands on
    the same point. Jaydari hashed the absolute file path, so moving the folder doubled the index.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'medvia-navi:{doc_id}:{chunk_index}'))


class VectorStore:
    """
    The only class that talks to Qdrant. Routes and services see this, never QdrantClient —
    so a later switch (MongoDB Atlas Vector Search, pgvector) changes one file.
    """

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._collection = settings.qdrant_collection
        self._dim = settings.embedding_dim
        # In-process Qdrant (tests, path:) has no payload indexes.
        self._local = (
            client is not None
            or settings.qdrant_url == ':memory:'
            or (settings.qdrant_url.startswith('path:'))
        )
        self._client = client or self._connect(settings)

    @staticmethod
    def _connect(settings: Settings) -> QdrantClient:
        if settings.qdrant_url == ':memory:':
            return QdrantClient(location=':memory:')
        if settings.qdrant_url.startswith('path:'):
            # Embedded Qdrant on local disk: no Docker, one process at a time (a file lock).
            return QdrantClient(path=settings.qdrant_url.removeprefix('path:'))
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout_s,
        )

    def ensure_collection(self) -> None:
        """
        Create the collection once; on later boots check that its vector size still matches
        the embedding model. A mismatch would make every search fail — better to refuse to start.
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            if not self._local:
                # Deleting "every chunk of doc X" filters on doc_id: index it on a server.
                self._client.create_payload_index(
                    self._collection, 'doc_id', field_schema=PayloadSchemaType.KEYWORD
                )
            logger.info('created collection %s (dim=%d)', self._collection, self._dim)
            return
        info = self._client.get_collection(self._collection)
        size = info.config.params.vectors.size  # type: ignore[union-attr]
        if size != self._dim:
            raise RuntimeError(
                f'collection {self._collection} has vectors of size {size}, '
                f'the embedding model gives {self._dim}: re-create the collection'
            )

    @staticmethod
    def _doc_filter(doc_id: str) -> Filter:
        return Filter(must=[FieldCondition(key='doc_id', match=MatchValue(value=doc_id))])

    def document_hash(self, doc_id: str) -> str | None:
        """The content hash stored with a document's chunks, or None when it is not indexed."""
        points, _ = self._client.scroll(
            self._collection, scroll_filter=self._doc_filter(doc_id), limit=1, with_payload=True
        )
        return (points[0].payload or {}).get('doc_hash') if points else None

    def replace_document(
        self, doc_id: str, doc_hash: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int:
        """
        Delete the document's old chunks, then write the new ones. Without the delete, a document
        that shrank from 5 chunks to 3 would keep chunks 3 and 4 — stale text answering forever.
        """
        self.delete_document(doc_id)
        points = [
            PointStruct(
                id=point_id(doc_id, chunk.chunk_index),
                vector=vector,
                payload={**chunk.model_dump(), 'doc_hash': doc_hash},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self._client.upsert(self._collection, points=points, wait=True)
        return len(points)

    def delete_document(self, doc_id: str) -> int:
        count = self.count(doc_id)
        if count:
            self._client.delete(self._collection, points_selector=self._doc_filter(doc_id))
        return count

    def count(self, doc_id: str | None = None) -> int:
        flt = self._doc_filter(doc_id) if doc_id else None
        return self._client.count(self._collection, count_filter=flt, exact=True).count

    def search(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        hits = self._client.query_points(
            self._collection, query=vector, limit=limit, with_payload=True
        ).points
        out = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload.pop('doc_hash', None)
            out.append(RetrievedChunk(**payload, score=float(hit.score)))
        return out

    def is_ready(self) -> bool:
        """For the readiness probe: True when Qdrant answers. Never raises."""
        try:
            self._client.get_collections()
            return True
        except Exception as err:  # noqa: BLE001 — any failure means "not ready"
            logger.warning('qdrant not ready: %s', err)
            return False

    def close(self) -> None:
        self._client.close()
