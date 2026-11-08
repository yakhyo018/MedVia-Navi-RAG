"""Ask Navi from the terminal, in-process: `python scripts/ask.py "question" [--adapter models/navi-lora]`."""

import argparse

from common import script_settings

from navi.container import build_container

parser = argparse.ArgumentParser()
parser.add_argument('questions', nargs='+')
parser.add_argument('--adapter', default=None)
parser.add_argument('--min-score', type=float, default=0.80)
args = parser.parse_args()

c = build_container(
    script_settings(generator='local', llm_adapter_path=args.adapter, min_score=args.min_score)
)
for q in args.questions:
    a = c.rag.answer(q)
    print(
        f'\nQ: {q}\nA: {a.answer}\n   refused={a.refused} cited={a.cited} invalid={a.invalid_citations} '
        f'llm={a.used_llm} {a.retrieval_ms}+{a.generation_ms} ms'
    )
    for i, s in enumerate(a.sources, 1):
        print(f'   [{i}] {s.score:.3f} {s.heading[:70]}')
c.store.close()
