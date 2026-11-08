"""
R5 — evaluate retrieval, the score threshold, and the base vs LoRA model on `data/dataset/test.jsonl`.

Everything runs as in production: the real retriever (multilingual-e5-small + Qdrant), the real
prompt, greedy decoding. Writes `reports/eval.json` and `reports/EVAL.md`.

Metrics
- Retrieval (answerable questions): Hit@1, Hit@4 — a supporting chunk is ranked first / in the
  four sources the model sees; MRR@10 (Mean Reciprocal Rank) — 1 / rank of the first one.
- Answer accuracy: the key fact (price, duration, hours…) is in the answer and it is not a refusal.
- Grounded citation: at least one cited source actually holds the fact.
- Language: the answer is in the question's language (a script / word heuristic, see `in_lang`).
- Refusal accuracy (unanswerable questions): the answer refuses instead of inventing.
- Invalid citations: a cited number that was not among the sources.

    python scripts/evaluate.py [--adapter models/navi-lora] [--min-score 0.80]
"""

import argparse
import gc
import json
import re
import statistics
from collections import defaultdict

from build_dataset import supports
from common import ROOT, script_settings

from navi.container import build_container
from navi.services.generator import LocalGenerator
from navi.services.rag import RagService

_CYR = re.compile('[а-яё]', re.I)
_UZ_WORDS = re.compile(
    r'(\b(va|narxi|turadi|daqiqa|mumkin|klinika\w*|ha|bilan|uchun|kerak|soat|ishlaydi|yopiq|'
    r"boshlanadi|davom|etadi|bo'ladi|qiling|emas|tashrif\w*)\b|yo'q|ma'lumot)",
    re.I,
)


def in_lang(answer: str, lang: str) -> bool:
    """RU: Cyrillic present. UZ: no Cyrillic and Uzbek function words. EN: neither."""
    if lang == 'RU':
        return bool(_CYR.search(answer))
    if lang == 'UZ':
        return not _CYR.search(answer) and bool(_UZ_WORDS.search(answer))
    return not _CYR.search(answer) and not _UZ_WORDS.search(answer)


def pct(xs: list[bool]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def retrieval_report(rag: RagService, tests: list[dict]) -> dict:
    rows = []
    for t in tests:
        hits = rag.retrieve(t['question'], 10)
        # An unanswerable question has no supporting chunk: only its top score is used.
        rank = None
        if t['answerable']:
            rank = next((i for i, h in enumerate(hits, 1) if supports(h, t)), None)
        rows.append((t, rank, hits[0].score if hits else 0.0))
    answerable = [r for r in rows if r[0]['answerable']]
    out = {'overall': summarise_ranks(answerable)}
    by_lang = defaultdict(list)
    for r in answerable:
        by_lang[r[0]['lang']].append(r)
    out['by_lang'] = {lang: summarise_ranks(rs) for lang, rs in sorted(by_lang.items())}
    # The threshold: does the top score alone separate answerable from unanswerable questions?
    pos = [s for t, _, s in rows if t['answerable']]
    neg = [s for t, _, s in rows if not t['answerable']]
    grid = []
    for th in [x / 100 for x in range(76, 95)]:
        kept = sum(s >= th for s in pos) / len(pos)
        blocked = sum(s < th for s in neg) / len(neg)
        grid.append(
            {
                'threshold': th,
                'answerable_kept': round(100 * kept, 1),
                'unanswerable_blocked': round(100 * blocked, 1),
            }
        )
    out['top1_score'] = {
        'answerable_mean': round(statistics.mean(pos), 3),
        'unanswerable_mean': round(statistics.mean(neg), 3),
        'answerable_min': round(min(pos), 3),
        'unanswerable_max': round(max(neg), 3),
    }
    out['threshold_grid'] = grid
    return out


def summarise_ranks(rows) -> dict:
    ranks = [r[1] for r in rows]
    return {
        'n': len(ranks),
        'hit@1': pct([r == 1 for r in ranks]),
        'hit@4': pct([r is not None and r <= 4 for r in ranks]),
        'mrr@10': round(sum(1 / r for r in ranks if r) / len(ranks), 3),
    }


def generation_report(rag: RagService, tests: list[dict]) -> tuple[dict, list[dict]]:
    records = []
    for t in tests:
        a = rag.answer(t['question'])
        rec = {
            'question': t['question'],
            'lang': t['lang'],
            'kind': t['kind'],
            'answerable': t['answerable'],
            'answer': a.answer,
            'refused': a.refused,
            'used_llm': a.used_llm,
            'cited': a.cited,
            'invalid_citations': a.invalid_citations,
            'ms': a.retrieval_ms + a.generation_ms,
        }
        if t['answerable']:
            supported = [i for i, s in enumerate(a.sources, 1) if supports(s, t)]
            rec['retrieved_support'] = bool(supported)
            rec['correct'] = t['key'].lower() in a.answer.lower() and not a.refused
            rec['grounded_citation'] = any(n in supported for n in a.cited)
            rec['in_lang'] = in_lang(a.answer, t['lang'])
        records.append(rec)
    return summarise(records), records


def summarise(records: list[dict]) -> dict:
    ans = [r for r in records if r['answerable']]
    una = [r for r in records if not r['answerable']]
    hit = [r for r in ans if r['retrieved_support']]
    out = {
        'answer_accuracy': pct([r['correct'] for r in ans]),
        'answer_accuracy_when_retrieved': pct([r['correct'] for r in hit]),
        'grounded_citation': pct([r['grounded_citation'] for r in ans]),
        'any_citation': pct([bool(r['cited']) for r in ans]),
        'answer_in_question_language': pct([r['in_lang'] for r in ans]),
        'false_refusal': pct([r['refused'] for r in hit]),
        'refusal_accuracy': pct([r['refused'] for r in una]),
        'invalid_citations': sum(len(r['invalid_citations']) for r in records),
        'mean_latency_ms': round(statistics.mean(r['ms'] for r in records)),
    }
    by_lang = {}
    for lang in sorted({r['lang'] for r in records}):
        a = [r for r in ans if r['lang'] == lang]
        u = [r for r in una if r['lang'] == lang]
        by_lang[lang] = {
            'answer_accuracy': pct([r['correct'] for r in a]),
            'grounded_citation': pct([r['grounded_citation'] for r in a]),
            'in_language': pct([r['in_lang'] for r in a]),
            'refusal_accuracy': pct([r['refused'] for r in u]),
        }
    out['by_lang'] = by_lang
    out['refusal_by_kind'] = {
        kind: pct([r['refused'] for r in una if r['kind'] == kind])
        for kind in sorted({r['kind'] for r in una})
    }
    return out


def free(rag: RagService) -> None:
    import torch

    del rag._generator.model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default=str(ROOT / 'models' / 'navi-lora'))
    parser.add_argument('--min-score', type=float, default=0.80)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    tests = [json.loads(line) for line in (ROOT / 'data' / 'dataset' / 'test.jsonl').open()]
    if args.limit:
        tests = tests[: args.limit]
    settings = script_settings(min_score=args.min_score)
    c = build_container(settings)
    report = {
        'test_questions': len(tests),
        'answerable': sum(t['answerable'] for t in tests),
        'unanswerable': sum(not t['answerable'] for t in tests),
        'min_score': args.min_score,
        'top_k': settings.top_k,
        # Unfiltered scores: rag.retrieve() drops hits below min_score, which would hide every
        # score under the threshold and make the grid below it meaningless.
        'retrieval': retrieval_report(
            RagService(c.store, c.embedder, None, top_k=settings.top_k, min_score=0.0), tests
        ),
    }
    print(json.dumps(report['retrieval']['overall']), flush=True)
    samples = {}
    for label, adapter in [('base', None), ('lora', args.adapter)]:
        gen = LocalGenerator(settings.llm_model, adapter_path=adapter, max_new_tokens=160)
        rag = RagService(c.store, c.embedder, gen, top_k=settings.top_k, min_score=args.min_score)
        report[label], samples[label] = generation_report(rag, tests)
        print(
            label,
            json.dumps({k: v for k, v in report[label].items() if not isinstance(v, dict)}),
            flush=True,
        )
        free(rag)
    c.store.close()

    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    (out / 'eval.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pairs = [
        {
            'question': b['question'],
            'lang': b['lang'],
            'answerable': b['answerable'],
            'base': b['answer'],
            'lora': lo['answer'],
        }
        for b, lo in zip(samples['base'], samples['lora'], strict=True)
    ]
    (out / 'eval_answers.jsonl').write_text(
        '\n'.join(json.dumps(p, ensure_ascii=False) for p in pairs) + '\n'
    )
    (out / 'EVAL.md').write_text(markdown(report, pairs))
    print('written reports/eval.json, reports/EVAL.md, reports/eval_answers.jsonl')


def markdown(r: dict, pairs: list[dict]) -> str:
    b, lo = r['base'], r['lora']
    rows = [
        ('Answer accuracy (key fact, answerable)', 'answer_accuracy'),
        ('… when retrieval found the source', 'answer_accuracy_when_retrieved'),
        ('Grounded citation', 'grounded_citation'),
        ('Any citation', 'any_citation'),
        ('Answer in the question language', 'answer_in_question_language'),
        ('False refusal (source was there)', 'false_refusal'),
        ('Refusal accuracy (unanswerable)', 'refusal_accuracy'),
        ('Invalid citations (count)', 'invalid_citations'),
        ('Mean latency, ms (M2, MPS)', 'mean_latency_ms'),
    ]
    lines = [
        '# MedVia Navi — evaluation',
        '',
        f'{r["test_questions"]} held-out questions ({r["answerable"]} answerable, {r["unanswerable"]} unanswerable), '
        f'top_k={r["top_k"]}, min_score={r["min_score"]}. Generated by `scripts/evaluate.py`.',
        '',
        '## Base vs LoRA',
        '',
        '| Metric | Qwen2.5-0.5B-Instruct | + Navi LoRA |',
        '| --- | ---: | ---: |',
        *[f'| {name} | {b[key]} | {lo[key]} |' for name, key in rows],
        '',
        '## By language',
        '',
        '| Lang | Accuracy base → LoRA | Grounded citation | In language | Refusal |',
        '| --- | --- | --- | --- | --- |',
    ]
    for lang in b['by_lang']:
        x, y = b['by_lang'][lang], lo['by_lang'][lang]
        lines.append(
            f'| {lang} | {x["answer_accuracy"]} → {y["answer_accuracy"]} | {x["grounded_citation"]} → {y["grounded_citation"]} '
            f'| {x["in_language"]} → {y["in_language"]} | {x["refusal_accuracy"]} → {y["refusal_accuracy"]} |'
        )
    ret = r['retrieval']
    lines += [
        '',
        '## Retrieval (multilingual-e5-small)',
        '',
        '| Set | n | Hit@1 | Hit@4 | MRR@10 |',
        '| --- | ---: | ---: | ---: | ---: |',
        f'| all | {ret["overall"]["n"]} | {ret["overall"]["hit@1"]} | {ret["overall"]["hit@4"]} | {ret["overall"]["mrr@10"]} |',
        *[
            f'| {lang} | {v["n"]} | {v["hit@1"]} | {v["hit@4"]} | {v["mrr@10"]} |'
            for lang, v in ret['by_lang'].items()
        ],
        '',
        f'Top-1 score: answerable mean {ret["top1_score"]["answerable_mean"]} (min {ret["top1_score"]["answerable_min"]}), '
        f'unanswerable mean {ret["top1_score"]["unanswerable_mean"]} (max {ret["top1_score"]["unanswerable_max"]}).',
        '',
        '| Threshold | Answerable kept % | Unanswerable blocked % |',
        '| ---: | ---: | ---: |',
        *[
            f'| {g["threshold"]:.2f} | {g["answerable_kept"]} | {g["unanswerable_blocked"]} |'
            for g in ret['threshold_grid'][::2]
        ],
        '',
        '## Sample answers',
        '',
    ]
    shown = 0
    for p in pairs:
        if shown >= 8:
            break
        if p['base'] != p['lora']:
            lines += [
                f'**{p["question"]}** ({p["lang"]}, {"answerable" if p["answerable"] else "unanswerable"})',
                '',
                f'- base: {p["base"]}',
                f'- LoRA: {p["lora"]}',
                '',
            ]
            shown += 1
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
