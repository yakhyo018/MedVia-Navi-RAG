import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from navi.core.security import require_api_key

router = APIRouter(prefix='/v1', tags=['query'], dependencies=[Depends(require_api_key)])

# One generation at a time: a 0.5B model on one GPU is not faster with two requests in
# parallel — both slow down and memory doubles. Others wait here instead of in the GPU.
_generation_slot = asyncio.Semaphore(1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceOut(BaseModel):
    index: int
    doc_id: str
    title: str
    heading: str
    text: str
    score: float
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    invalid_citations: list[int]
    used_llm: bool
    retrieval_ms: int
    generation_ms: int


@router.post('/query', response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """
    The model call blocks for seconds: it runs in a worker thread (run_in_threadpool) so the
    event loop keeps answering /health and other requests meanwhile. Jaydari's sync `def`
    handlers had no limit — ten parallel questions meant ten generations fighting for one GPU.
    """
    rag = request.app.state.container.rag
    async with _generation_slot:
        result = await run_in_threadpool(rag.answer, payload.question, payload.top_k)
    return QueryResponse(
        answer=result.answer,
        refused=result.refused,
        sources=[
            SourceOut(
                index=i,
                doc_id=s.doc_id,
                title=s.title,
                heading=s.heading,
                text=s.text,
                score=round(s.score, 4),
                cited=i in result.cited,
            )
            for i, s in enumerate(result.sources, start=1)
        ],
        invalid_citations=result.invalid_citations,
        used_llm=result.used_llm,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
    )
