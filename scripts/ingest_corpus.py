"""Index the whole corpus into the local Qdrant (`qdrant_local/`). Unchanged documents are skipped."""

import time

from common import load_corpus, script_settings

from navi.container import build_container


def main() -> None:
    t0 = time.perf_counter()
    c = build_container(script_settings())
    counts: dict[str, int] = {}
    chunks = 0
    for doc in load_corpus():
        r = c.ingest.ingest(doc)
        counts[r.status] = counts.get(r.status, 0) + 1
        chunks += r.chunks
    print(
        f'{sum(counts.values())} documents {counts}, {chunks} chunks, {time.perf_counter() - t0:.1f}s'
    )
    c.store.close()


if __name__ == '__main__':
    main()
