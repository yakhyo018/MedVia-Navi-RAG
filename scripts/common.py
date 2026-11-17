"""Shared by the scripts: load the corpus, build services without the HTTP layer."""

import json
import os
from pathlib import Path

from navi.core.config import Settings
from navi.domain import Document

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'data' / 'corpus'
# Embedded Qdrant on disk: the scripts run without Docker and without the API.
LOCAL_QDRANT = f'path:{ROOT / "qdrant_local"}'


def load_guides() -> list[Document]:
    """The hand-written guides (`guide-*`): the only content MedVia's nightly sync never sends."""
    docs = []
    for md in sorted((CORPUS / 'guides').glob('*.md')):
        text = md.read_text(encoding='utf-8')
        title = text.splitlines()[0].lstrip('# ').strip()
        docs.append(
            Document(doc_id=f'guide-{md.stem}', title=title, text=text, source_type='guide')
        )
    return docs


def load_corpus() -> list[Document]:
    docs = [Document(**json.loads(line)) for line in (CORPUS / 'medvia.jsonl').open()]
    return docs + load_guides()


def script_settings(**overrides) -> Settings:
    """A key is required by Settings; the scripts never serve HTTP, so a throwaway one is fine."""
    base = {
        'api_key': os.environ.get('NAVI_API_KEY') or 'script-only-' + 'x' * 32,
        'qdrant_url': LOCAL_QDRANT,
        'generator': 'fake',
        '_env_file': None,
    }
    return Settings(**{**base, **overrides})
