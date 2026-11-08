import time
from dataclasses import dataclass, field

from navi.domain import RetrievedChunk
from navi.services.embedder import Embedder
from navi.services.generator import Generator
from navi.services.prompting import build_messages, citations, is_refusal, refusal_for
from navi.services.vector_store import VectorStore


@dataclass
class Answer:
    answer: str
    refused: bool
    sources: list[RetrievedChunk]
    cited: list[int]
    # Numbers the model cited that were not among the sources — a hallucinated citation.
    invalid_citations: list[int] = field(default_factory=list)
    used_llm: bool = True
    retrieval_ms: int = 0
    generation_ms: int = 0


class RagService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        generator: Generator,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._generator = generator
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        hits = self._store.search(self._embedder.embed_query(question), top_k or self.top_k)
        return [h for h in hits if h.score >= self.min_score]

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        """
        Retrieval decides first: nothing above min_score → the refusal, without calling the model
        at all. Cheaper, and a model that is never asked cannot invent an answer. Jaydari's
        keyword list ("file", "api"…) decided by words in the question; a similarity threshold
        decides by what the index actually holds.
        """
        t0 = time.perf_counter()
        sources = self.retrieve(question, top_k)
        t1 = time.perf_counter()
        if not sources:
            return Answer(
                answer=refusal_for(question),
                refused=True,
                sources=[],
                cited=[],
                used_llm=False,
                retrieval_ms=round((t1 - t0) * 1000),
            )
        text = self._generator.generate(build_messages(question, sources))
        t2 = time.perf_counter()
        cited = citations(text)
        valid = [n for n in cited if 1 <= n <= len(sources)]
        return Answer(
            answer=text,
            refused=is_refusal(text),
            sources=sources,
            cited=valid,
            invalid_citations=[n for n in cited if n not in valid],
            retrieval_ms=round((t1 - t0) * 1000),
            generation_ms=round((t2 - t1) * 1000),
        )
