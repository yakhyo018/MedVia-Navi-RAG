"""
Send only the hand-written guides (`data/corpus/guides`, ids `guide-*`) to a running Navi API —
in production, from inside its own container:

    docker exec medvia-navi python scripts/ingest_guides.py

Through the API (`POST /v1/documents`, the same door MedVia's batch uses), not in-process: a
second process would load e5 a second time next to the API's own models and could push the
container over its memory limit.

MedVia's own content (clinics, treatments, tips, FAQ, notices) is not here: medvia-batch's
NAVI_SYNC sends it from the production database. `data/corpus/medvia.jsonl` is an export of the
development database — its ids do not exist in production, so it is never ingested there.
Unchanged guides are skipped, so running this twice is harmless.
"""

import json
import os
import urllib.request

from common import load_guides


def main() -> None:
    url = os.environ.get('NAVI_URL', 'http://127.0.0.1:8100').rstrip('/') + '/v1/documents'
    # Only http(s): urllib would also open file:// — S310 below is checked here.
    if not url.startswith(('http://', 'https://')):
        raise SystemExit(f'NAVI_URL must be http(s), got {url}')
    body = json.dumps({'documents': [d.model_dump() for d in load_guides()]}).encode()
    request = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        method='POST',
        headers={'content-type': 'application/json', 'x-api-key': os.environ['NAVI_API_KEY']},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        items = json.load(response)['items']
    counts: dict[str, int] = {}
    for item in items:
        counts[item['status']] = counts.get(item['status'], 0) + 1
        print(f'{item["doc_id"]}: {item["status"]} ({item["chunks"]} chunks)')
    print(f'{len(items)} guides {counts}')


if __name__ == '__main__':
    main()
