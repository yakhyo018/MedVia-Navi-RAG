"""
Index only the hand-written guides (`data/corpus/guides`, ids `guide-*`) into the Qdrant that
the environment points at — the production server's, inside the container:

    docker exec medvia-navi python scripts/ingest_guides.py

MedVia's own content (clinics, treatments, tips, FAQ, notices) is not here: medvia-batch's
NAVI_SYNC sends it from the production database. `data/corpus/medvia.jsonl` is an export of the
development database — its ids do not exist in production, so it is never ingested there.
Unchanged guides are skipped, so running this twice is harmless.
"""

from common import load_guides

from navi.container import build_container
from navi.core.config import Settings


def main() -> None:
    # Settings from the environment (NAVI_API_KEY, NAVI_QDRANT_URL); no LLM is needed to ingest.
    c = build_container(Settings(generator='fake'))
    counts: dict[str, int] = {}
    for doc in load_guides():
        r = c.ingest.ingest(doc)
        counts[r.status] = counts.get(r.status, 0) + 1
        print(f'{doc.doc_id}: {r.status} ({r.chunks} chunks)')
    print(f'{sum(counts.values())} guides {counts}')
    c.store.close()


if __name__ == '__main__':
    main()
