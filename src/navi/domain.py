"""The service's own data shapes — shared by the API, the services and the scripts."""

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal['clinic', 'treatment', 'health_tip', 'faq', 'notice', 'guide']


class Document(BaseModel):
    """
    One piece of MedVia content. `doc_id` is MedVia's own id (`clinic-<mongoId>`, `guide-<slug>`):
    sending the same id again replaces the document — its old chunks are removed first.
    """

    doc_id: str = Field(min_length=1, max_length=120, pattern=r'^[a-z0-9][a-z0-9_\-]*$')
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=50_000)
    source_type: SourceType
    lang: str = Field(default='EN', pattern='^(KO|EN|RU|UZ|KK|ZH)$')
    meta: dict[str, str | int | float | None] = Field(default_factory=dict)


class Chunk(BaseModel):
    doc_id: str
    chunk_index: int
    title: str
    # "Title › Section" — the heading path, so a chunk still says what it is about.
    heading: str
    text: str
    source_type: SourceType


class RetrievedChunk(Chunk):
    score: float
