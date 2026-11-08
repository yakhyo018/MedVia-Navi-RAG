"""Builds the services from Settings — one place that knows which implementation runs."""

from dataclasses import dataclass

from navi.core.config import Settings
from navi.services.embedder import E5Embedder, Embedder, HashEmbedder
from navi.services.generator import FakeGenerator, Generator, LocalGenerator
from navi.services.ingest import IngestService
from navi.services.rag import RagService
from navi.services.vector_store import VectorStore


@dataclass
class Container:
    store: VectorStore
    embedder: Embedder
    generator: Generator
    ingest: IngestService
    rag: RagService


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == 'hash':
        return HashEmbedder(settings.embedding_dim)
    return E5Embedder(settings.embedding_model, device='cpu')


def build_generator(settings: Settings) -> Generator:
    if settings.generator == 'fake':
        return FakeGenerator()
    return LocalGenerator(
        settings.llm_model,
        adapter_path=settings.llm_adapter_path or None,
        device=settings.device,
        max_new_tokens=settings.max_new_tokens,
    )


def build_container(
    settings: Settings,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
) -> Container:
    store = store or VectorStore(settings)
    store.ensure_collection()
    embedder = embedder or build_embedder(settings)
    if embedder.dim != settings.embedding_dim:
        raise RuntimeError(
            f'NAVI_EMBEDDING_DIM={settings.embedding_dim}, the model gives {embedder.dim}'
        )
    generator = generator or build_generator(settings)
    return Container(
        store=store,
        embedder=embedder,
        generator=generator,
        ingest=IngestService(store, embedder),
        rag=RagService(
            store, embedder, generator, top_k=settings.top_k, min_score=settings.min_score
        ),
    )
