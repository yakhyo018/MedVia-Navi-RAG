"""
Markdown → chunks, split by headings first and by size only when a section is too long.

Jaydari cut every 1,200 characters, wherever that fell — mid-sentence, a price separated from
its treatment. A heading is where the author already split the meaning, so that is the first
cut. Every chunk carries its heading path ("Seoul Glow › Opening hours"): the words of the
heading are usually what the question asks about, and they would be lost in a bare section.
"""

import re

from navi.domain import Chunk, Document

# Part of every document's hash: a new chunking rule must re-index everything.
CHUNKER_VERSION = 'md-headings-v1'

_HEADING = re.compile(r'^(#{1,6})\s+(.+?)\s*$')


def _sections(text: str) -> list[tuple[list[str], str]]:
    """(heading path, body) for every section; text before the first heading has an empty path."""
    path: list[str] = []
    body: list[str] = []
    out: list[tuple[list[str], str]] = []

    def flush() -> None:
        content = '\n'.join(body).strip()
        if content:
            out.append((list(path), content))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2)]
            continue
        body.append(line)
    flush()
    return out


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Paragraph by paragraph; a single paragraph longer than max_chars is cut with overlap."""
    pieces: list[str] = []
    current = ''
    for para in [p.strip() for p in text.split('\n\n') if p.strip()]:
        if len(current) + len(para) + 2 <= max_chars:
            current = f'{current}\n\n{para}' if current else para
            continue
        if current:
            pieces.append(current)
        if len(para) <= max_chars:
            current = para
            continue
        start = 0
        while start < len(para):
            pieces.append(para[start : start + max_chars])
            if start + max_chars >= len(para):
                break
            start += max_chars - overlap
        current = ''
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: Document, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, body in _sections(doc.text):
        # The H1 ("Seoul Glow Dermatology (서울글로우피부과)") repeats the title: drop it.
        heading_parts = [doc.title] + [p for p in path if not p.startswith(doc.title)]
        heading = ' › '.join(heading_parts)
        for piece in _split_long(body, max_chars, overlap):
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_index=len(chunks),
                    title=doc.title,
                    heading=heading,
                    text=piece,
                    source_type=doc.source_type,
                )
            )
    return chunks
