"""
Build the fine-tuning and evaluation sets: `data/dataset/train.jsonl`, `data/dataset/test.jsonl`.

What the LoRA should learn is a behaviour, not facts (the facts come from retrieval):
cite the source that holds the fact, answer in the question's language, refuse when the
sources do not hold the answer. So every training prompt is built exactly like a live one —
the real retriever picks the sources, `build_messages` renders them — and the target answer
cites the right number.

Questions come from three places:
- templates over the structured content (treatment prices and durations, clinic hours,
  addresses, interpreters) in EN / RU / UZ;
- hand-written questions about the FAQ, health tips, notices and guides (`data/qa/manual.jsonl`);
- questions the content cannot answer (`data/qa/unanswerable.jsonl`, plus "a treatment at a
  clinic that does not offer it"), and hard negatives: an answerable question whose supporting
  chunks were taken out of the sources.

Split: template questions by document (a quarter of the clinics and treatments are never seen
in training — the test asks about unseen documents); hand-written ones by position.
"""

import hashlib
import json
import random
import re
from collections import Counter

from common import ROOT, load_corpus, script_settings

from navi.container import build_container
from navi.domain import Document, RetrievedChunk
from navi.services.chunker import chunk_document
from navi.services.prompting import REFUSALS, build_messages

TOP_K = 4
OUT = ROOT / 'data' / 'dataset'
QA = ROOT / 'data' / 'qa'

LANG_EN = {'KO': 'Korean', 'EN': 'English', 'RU': 'Russian', 'UZ': 'Uzbek', 'KK': 'Kazakh'}
LANG_RU = {
    'KO': 'корейском',
    'EN': 'английском',
    'RU': 'русском',
    'UZ': 'узбекском',
    'KK': 'казахском',
}


def stable(text: str) -> int:
    return int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8], 16)


def is_test_doc(doc_id: str) -> bool:
    return stable(doc_id) % 4 == 0


def join(items: list[str], conj: str) -> str:
    return items[0] if len(items) == 1 else f'{", ".join(items[:-1])} {conj} {items[-1]}'


def line(text: str, prefix: str) -> str | None:
    for ln in text.splitlines():
        if ln.strip().startswith(prefix):
            return ln.strip()[len(prefix) :].strip()
    return None


# Hand-written answers in RU / UZ carry their key fact in that language ('напрямую'); the
# English source says it differently ('directly'). The source key is what a chunk must contain.
SOURCE_KEY = {
    'напрямую': 'directly',
    'klinika': 'directly',
    'без дополнительной платы': 'no extra charge',
    "to'lovsiz": 'no extra charge',
    'врач': 'doctor',
    'sakkiz': 'eight',
    'шесть': 'six months',
    'штраф': 'no cancellation fee',
    'аллерги': 'allergies',
    "o'zbek": 'Uzbek',
    'паспорт': 'passport',
}


def item(question, lang, answer, key, doc_id, entity=None, kind='template', source_key=None):
    """`key` must appear in a correct answer; `source_key` in a chunk that supports it."""
    return {
        'question': question,
        'lang': lang,
        'answerable': True,
        'answer': answer,
        'key': key,
        'source_key': source_key or key,
        'gold_doc': doc_id,
        'entity': entity,
        'kind': kind,
    }


def treatment_questions(doc: Document) -> list[dict]:
    t, c = doc.meta['treatmentName'], doc.meta['clinicName']
    p, d = f'{int(doc.meta["price"]):,}', str(doc.meta['duration'])
    q = []
    q.append(
        item(
            f'How much does {t} cost at {c}?',
            'EN',
            f'{t} at {c} costs from {p} KRW.',
            p,
            doc.doc_id,
            c,
        )
    )
    q.append(
        item(
            f'How long does {t} take at {c}?',
            'EN',
            f'{t} at {c} takes {d} minutes.',
            d,
            doc.doc_id,
            c,
        )
    )
    q.append(
        item(f'Сколько стоит {t} в {c}?', 'RU', f'{t} в {c} стоит от {p} вон.', p, doc.doc_id, c)
    )
    q.append(
        item(f'Сколько длится {t} в {c}?', 'RU', f'{t} в {c} длится {d} минут.', d, doc.doc_id, c)
    )
    q.append(
        item(
            f'{c} klinikasida {t} qancha turadi?',
            'UZ',
            f'{c} klinikasida {t} narxi {p} vondan boshlanadi.',
            p,
            doc.doc_id,
            c,
        )
    )
    q.append(
        item(
            f'{c} klinikasida {t} qancha davom etadi?',
            'UZ',
            f'{c} klinikasida {t} {d} daqiqa davom etadi.',
            d,
            doc.doc_id,
            c,
        )
    )
    return q


def clinic_questions(doc: Document) -> list[dict]:
    c, text = doc.title, doc.text
    q = []
    address = re.search(r'Address: (.+?)\. Phone: (.+?)\.', text)
    if address:
        addr, phone = address.group(1), address.group(2)
        street = ' '.join(addr.split(',')[0].split()[:2])
        q.append(
            item(f'What is the address of {c}?', 'EN', f'{c} is at {addr}.', street, doc.doc_id, c)
        )
        q.append(item(f'Какой адрес у {c}?', 'RU', f'Адрес {c}: {addr}.', street, doc.doc_id, c))
        q.append(
            item(
                f'What is the phone number of {c}?',
                'EN',
                f'The phone number of {c} is {phone}.',
                phone,
                doc.doc_id,
                c,
            )
        )
    sunday = line(text, '- Sunday:')
    if sunday:
        if sunday == 'closed':
            q.append(
                item(
                    f'Is {c} open on Sunday?',
                    'EN',
                    f'No, {c} is closed on Sunday.',
                    'closed',
                    doc.doc_id,
                    c,
                    source_key='Sunday: closed',
                )
            )
            q.append(
                item(
                    f'Работает ли {c} в воскресенье?',
                    'RU',
                    f'Нет, в воскресенье {c} не работает.',
                    'не работает',
                    doc.doc_id,
                    c,
                    source_key='Sunday: closed',
                )
            )
            q.append(
                item(
                    f'{c} yakshanba kuni ishlaydimi?',
                    'UZ',
                    f"Yo'q, {c} yakshanba kuni yopiq.",
                    'yopiq',
                    doc.doc_id,
                    c,
                    source_key='Sunday: closed',
                )
            )
        else:
            open_at = sunday.split('–')[0]
            q.append(
                item(
                    f'Is {c} open on Sunday?',
                    'EN',
                    f'Yes, on Sunday {c} is open {sunday}.',
                    open_at,
                    doc.doc_id,
                    c,
                )
            )
            q.append(
                item(
                    f'Работает ли {c} в воскресенье?',
                    'RU',
                    f'Да, в воскресенье {c} работает {sunday}.',
                    open_at,
                    doc.doc_id,
                    c,
                )
            )
            q.append(
                item(
                    f'{c} yakshanba kuni ishlaydimi?',
                    'UZ',
                    f'Ha, {c} yakshanba kuni {sunday} ishlaydi.',
                    open_at,
                    doc.doc_id,
                    c,
                )
            )
    saturday = line(text, '- Saturday:')
    if saturday and saturday != 'closed':
        q.append(
            item(
                f'What are the Saturday hours of {c}?',
                'EN',
                f'On Saturday {c} is open {saturday}.',
                saturday.split('–')[0],
                doc.doc_id,
                c,
            )
        )
    interp = line(text, 'Consultations in:')
    # Mintleaf's description names three interpreter languages, its structured field one: the
    # demo seed contradicts itself there, so no question is asked about it (see the report).
    if interp and 'Interpreter languages:' in interp and 'Mintleaf' not in c:
        codes = [
            x.strip() for x in interp.split('Interpreter languages:')[1].strip(' .').split(',')
        ]
        codes = [x for x in codes if x in LANG_EN]
        if codes:
            en = join([LANG_EN[x] for x in codes], 'and')
            ru = join([LANG_RU[x] for x in codes], 'и')
            q.append(
                item(
                    f'Which languages does {c} offer interpreters in?',
                    'EN',
                    f'{c} provides interpreters in {en}.',
                    LANG_EN[codes[0]],
                    doc.doc_id,
                    c,
                    source_key='Interpreter languages:',
                )
            )
            q.append(
                item(
                    f'На каких языках в {c} есть переводчик?',
                    'RU',
                    f'В {c} есть переводчик на {ru} {"языке" if len(codes) == 1 else "языках"}.',
                    LANG_RU[codes[0]],
                    doc.doc_id,
                    c,
                    source_key='Interpreter languages:',
                )
            )
    return q


def with_citation(answer: str, n: int) -> str:
    return f'{answer.rstrip(".")} [{n}].'


def supports(chunk: RetrievedChunk, it: dict) -> bool:
    """A chunk supports the answer when it holds the key fact about the right entity."""
    text = f'{chunk.heading}\n{chunk.text}'.lower()
    if it['source_key'].lower() not in text:
        return False
    if chunk.doc_id == it['gold_doc']:
        return True
    return bool(it.get('entity')) and it['entity'].lower() in text


def main() -> None:
    docs = load_corpus()
    by_id = {d.doc_id: d for d in docs}
    by_title = {d.title: d for d in docs}
    chunks = {d.doc_id: chunk_document(d) for d in docs}
    c = build_container(script_settings(min_score=0.0))

    answerable: list[dict] = []
    for d in docs:
        if d.source_type == 'treatment':
            answerable += treatment_questions(d)
        elif d.source_type == 'clinic':
            answerable += clinic_questions(d)
    for split_pos, line_ in enumerate((QA / 'manual.jsonl').open()):
        m = json.loads(line_)
        doc = by_title[m['doc_title']]
        source_key = SOURCE_KEY.get(m['key'], m['key'])
        if source_key.lower() not in doc.text.lower():
            raise SystemExit(f'{source_key!r} is not in {doc.title!r}')
        it = item(
            m['question'],
            m['lang'],
            m['answer'],
            m['key'],
            doc.doc_id,
            kind='manual',
            source_key=source_key,
        )
        it['test'] = split_pos % 4 == 3
        answerable.append(it)
    for it in answerable:
        it.setdefault('test', is_test_doc(it['gold_doc']))

    unanswerable = []
    for pos, line_ in enumerate((QA / 'unanswerable.jsonl').open()):
        u = json.loads(line_)
        unanswerable.append({**u, 'answerable': False, 'kind': 'manual', 'test': pos % 4 == 3})
    # "A treatment at a clinic that does not offer it" — retrieval brings the same treatment at
    # another clinic, with a price: the tempting wrong answer.
    offered: dict[str, set[str]] = {}
    for d in docs:
        if d.source_type == 'treatment':
            offered.setdefault(d.meta['clinicName'], set()).add(d.meta['treatmentName'])
    all_treatments = sorted({t for ts in offered.values() for t in ts})
    for clinic, ts in sorted(offered.items()):
        rng = random.Random(stable(clinic))
        for t in rng.sample([x for x in all_treatments if x not in ts], 2):
            test = stable(f'{clinic}:{t}') % 4 == 0
            # Half price, half duration. v1 asked only the price: 38 of 110 price questions in
            # train ended in a refusal, and the LoRA learnt "price question → refuse" (10 % false
            # refusals). The form of the question must not predict the answer — only the sources.
            price = stable(f'{t}:{clinic}') % 2 == 0
            unanswerable.append(
                {
                    'question': f'How much does {t} cost at {clinic}?'
                    if price
                    else f'How long does {t} take at {clinic}?',
                    'lang': 'EN',
                    'answerable': False,
                    'kind': 'not_offered',
                    'test': test,
                }
            )
            unanswerable.append(
                {
                    'question': f'Сколько стоит {t} в {clinic}?'
                    if price
                    else f'Сколько длится {t} в {clinic}?',
                    'lang': 'RU',
                    'answerable': False,
                    'kind': 'not_offered',
                    'test': test,
                }
            )

    train, test = [], []
    for it in answerable:
        if it['test']:
            test.append({k: v for k, v in it.items() if k != 'test'})
            continue
        rng = random.Random(stable(it['question']))
        hits = c.rag.retrieve(it['question'], TOP_K * 3)
        sources = hits[:TOP_K]
        support = [h for h in sources if supports(h, it)]
        if not support:
            # Training only: put the supporting chunk in, so the target can cite something real.
            gold = next(
                (
                    ch
                    for ch in chunks[it['gold_doc']]
                    if it['source_key'].lower() in f'{ch.heading}\n{ch.text}'.lower()
                ),
                chunks[it['gold_doc']][0],
            )
            sources[-1] = RetrievedChunk(**gold.model_dump(), score=0.0)
        rng.shuffle(sources)
        n = next((i for i, s in enumerate(sources, 1) if supports(s, it)), None)
        if n is None:
            raise SystemExit(
                f'key {it["key"]!r} is in no chunk of {it["gold_doc"]}: {it["question"]}'
            )
        train.append(example(it, sources, with_citation(it['answer'], n)))
        # Hard negative for a fifth of them: the same question, every supporting chunk removed.
        if rng.random() < 0.2:
            negatives = [h for h in hits if it['source_key'].lower() not in h.text.lower()]
            if len(negatives) >= TOP_K:
                neg = {**it, 'answerable': False, 'kind': 'hard_negative'}
                train.append(example(neg, negatives[:TOP_K], REFUSALS[it['lang']]))

    for u in unanswerable:
        if u['test']:
            test.append({k: v for k, v in u.items() if k != 'test'})
            continue
        sources = c.rag.retrieve(u['question'], TOP_K)
        train.append(example(u, sources, REFUSALS[u['lang']]))

    random.Random(7).shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'train.jsonl', train)
    write(OUT / 'test.jsonl', test)
    print(
        f'train {len(train)}:',
        dict(Counter((e['meta']['kind'], e['meta']['answerable']) for e in train)),
    )
    print(f'test  {len(test)}:', dict(Counter((e['kind'], e['answerable']) for e in test)))
    print('test langs:', dict(Counter(e['lang'] for e in test)))
    c.store.close()
    del by_id


def example(it: dict, sources: list[RetrievedChunk], target: str) -> dict:
    """TRL's prompt–completion format: the loss is computed on the completion only."""
    return {
        'prompt': build_messages(it['question'], sources),
        'completion': [{'role': 'assistant', 'content': target}],
        'meta': {
            k: it.get(k)
            for k in ('question', 'lang', 'answerable', 'kind', 'key', 'source_key', 'gold_doc')
        },
    }


def write(path, rows) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
