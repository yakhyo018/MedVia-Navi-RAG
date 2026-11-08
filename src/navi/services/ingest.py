import hashlib
from dataclasses import dataclass

from navi.domain import Document
from navi.services.chunker import CHUNKER_VERSION, chunk_document
from navi.services.embedder import Embedder
from navi.services.vector_store import VectorStore


@dataclass
class IngestResult:
    doc_id: str
    status: str  # 'indexed' | 'unchanged'
    chunks: int


def document_hash(doc: Document) -> str:
    """
    SHA-256 of everything that decides the index: the text and title, and the chunker version —
    without it, a new chunking rule would leave every old document "unchanged" and un-rechunked.
    """
    raw = '\x1f'.join([CHUNKER_VERSION, doc.title, doc.source_type, doc.lang, doc.text])
    return hashlib.sha256(raw.encode()).hexdigest()


class IngestService:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def ingest(self, doc: Document) -> IngestResult:
        """
        Unchanged documents are skipped: MedVia's batch may send the whole catalogue every night,
        and embedding is the slow part. The hash decides, not a timestamp — a re-sent document
        with the same text costs one lookup.
        """
        digest = document_hash(doc)
        if self._store.document_hash(doc.doc_id) == digest:
            return IngestResult(doc.doc_id, 'unchanged', self._store.count(doc.doc_id))
        chunks = chunk_document(doc)
        # The heading goes into the vector too: "Seoul Glow › Opening hours" + the hours.
        vectors = self._embedder.embed_passages([f'{c.heading}\n{c.text}' for c in chunks])
        written = self._store.replace_document(doc.doc_id, digest, chunks, vectors)
        return IngestResult(doc.doc_id, 'indexed', written)

    def delete(self, doc_id: str) -> int:
        return self._store.delete_document(doc_id)
