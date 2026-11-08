from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from navi.core.security import require_api_key
from navi.domain import Document

router = APIRouter(
    prefix='/v1/documents', tags=['documents'], dependencies=[Depends(require_api_key)]
)


class IngestRequest(BaseModel):
    # 200 per call: one request stays well under Qdrant's 32 MB body limit and a timeout.
    documents: list[Document] = Field(min_length=1, max_length=200)


class IngestItem(BaseModel):
    doc_id: str
    status: str
    chunks: int


class IngestResponse(BaseModel):
    items: list[IngestItem]


class DocumentInfo(BaseModel):
    doc_id: str
    chunks: int


@router.post('', response_model=IngestResponse)
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    """
    Documents come in the body — never a server path. Jaydari's /ingest-path read any folder on
    the server ({"path": "~"} → .env and SSH keys into the index, and back out as "sources").
    """
    ids = [d.doc_id for d in payload.documents]
    if len(ids) != len(set(ids)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, 'doc_id must be unique')
    service = request.app.state.container.ingest
    items = [IngestItem(**vars(service.ingest(d))) for d in payload.documents]
    return IngestResponse(items=items)


@router.get('/{doc_id}', response_model=DocumentInfo)
def info(doc_id: str, request: Request) -> DocumentInfo:
    chunks = request.app.state.container.store.count(doc_id)
    if not chunks:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'document not found')
    return DocumentInfo(doc_id=doc_id, chunks=chunks)


@router.delete('/{doc_id}', response_model=DocumentInfo)
def delete(doc_id: str, request: Request) -> DocumentInfo:
    """MedVia deletes a clinic or a tip → its chunks go, or Navi keeps answering from it."""
    removed = request.app.state.container.ingest.delete(doc_id)
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'document not found')
    return DocumentInfo(doc_id=doc_id, chunks=removed)
