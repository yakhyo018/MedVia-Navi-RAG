"""
Export MedVia's public content into Navi documents: `data/corpus/medvia.jsonl`.

Reads only MedVia's public GraphQL API (what any visitor of the site can see) — Navi never
connects to MedVia's database. The demo data is synthetic (fictional clinics and doctors).

    python scripts/export_medvia.py --api http://localhost:3000/graphql
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

# Hand-typed test posts in a developer's database ('test', 'dfdsfs') are not knowledge.
MIN_CONTENT_CHARS = 30

DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def gql(api: str, query: str, variables: dict | None = None) -> dict:
    body = json.dumps({'query': query, 'variables': variables or {}}).encode()
    req = urllib.request.Request(api, body, {'content-type': 'application/json'})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=30) as res:  # noqa: S310 — our own URL
            data = json.load(res)
    except urllib.error.HTTPError as err:  # GraphQL validation errors come back as 400
        raise RuntimeError(err.read().decode()[:500]) from err
    if data.get('errors'):
        raise RuntimeError(json.dumps(data['errors']))
    return data['data']


def won(price: int) -> str:
    return f'{price:,} KRW'


def hours_text(hours: list[dict]) -> str:
    by_day = {h['day']: h for h in hours}
    lines = []
    for day in [1, 2, 3, 4, 5, 6, 0]:
        h = by_day.get(day)
        if not h or h.get('closed'):
            lines.append(f'- {DAYS[day]}: closed')
        else:
            lines.append(f'- {DAYS[day]}: {h["open"]}–{h["close"]}')
    return '\n'.join(lines)


CLINICS_Q = """query($i: ClinicsInquiry!) { getClinics(input: $i) { list {
  _id clinicName clinicNameKo clinicSpecialty clinicDistrict clinicAddress clinicPhone
  clinicHours { day open close closed } clinicLangs clinicInterpreterLangs clinicDesc
  clinicRating clinicReviews } } }"""
TREATMENTS_Q = """query($i: ClinicTreatmentsInquiry!) { getClinicTreatments(input: $i) {
  _id treatmentName treatmentCategory treatmentDuration treatmentPrice treatmentPriceNote
  treatmentDesc treatmentAftercare treatmentLangs doctors { doctorName } } }"""
DOCTORS_Q = """query($i: DoctorsInquiry!) { getDoctors(input: $i) { list {
  _id doctorName doctorNameKo doctorExperience doctorLangs doctorBio } } }"""
ARTICLES_Q = """query($i: ArticlesInquiry!) { getArticles(input: $i) { list {
  _id articleTitle articleContent articleLang authorName } } }"""
NOTICES_Q = """query($i: NoticesInquiry!) { getNotices(input: $i) { list {
  _id noticeCategory noticeFaqCategory noticeTitle noticeContent } } }"""


def clinic_documents(api: str) -> list[dict]:
    docs = []
    clinics = gql(api, CLINICS_Q, {'i': {'page': 1, 'limit': 50}})['getClinics']['list']
    for c in clinics:
        treatments = gql(api, TREATMENTS_Q, {'i': {'clinicId': c['_id']}})['getClinicTreatments']
        doctors = gql(
            api, DOCTORS_Q, {'i': {'page': 1, 'limit': 20, 'search': {'clinicId': c['_id']}}}
        )['getDoctors']['list']
        name = c['clinicName']
        langs = ', '.join(c['clinicLangs'])
        interp = ', '.join(c.get('clinicInterpreterLangs') or []) or 'none'
        text = (
            f'# {name} ({c["clinicNameKo"]})\n\n'
            f'{name} is a {c["clinicSpecialty"].replace("_", " ").lower()} clinic in '
            f'{c["clinicDistrict"].title()}-gu, Seoul. Address: {c["clinicAddress"]}. '
            f'Phone: {c["clinicPhone"]}.\n\n{c["clinicDesc"]}\n\n'
            f'## Languages\n\nConsultations in: {langs}. Interpreter languages: {interp}.\n\n'
            f'## Opening hours (Seoul time)\n\n{hours_text(c["clinicHours"])}\n\n'
            f'## Doctors\n\n'
            + '\n'.join(
                f'- {d["doctorName"]} ({d["doctorNameKo"]}), {d["doctorExperience"]} years of '
                f'experience. {d["doctorBio"]}'
                for d in doctors
            )
            + '\n\n## Treatments\n\n'
            + '\n'.join(
                f'- {t["treatmentName"]}: from {won(t["treatmentPrice"])}, '
                f'{t["treatmentDuration"]} minutes.'
                for t in treatments
            )
        )
        docs.append(
            {
                'doc_id': f'clinic-{c["_id"]}',
                'title': name,
                'source_type': 'clinic',
                'lang': 'EN',
                'text': text,
                'meta': {'clinicId': c['_id'], 'clinicName': name, 'district': c['clinicDistrict']},
            }
        )
        for t in treatments:
            doctors_of = ', '.join(d['doctorName'] for d in t.get('doctors') or []) or 'any doctor'
            body = (
                f'# {t["treatmentName"]} at {name}\n\n'
                f'{t["treatmentName"]} at {name} ({c["clinicDistrict"].title()}-gu) costs from '
                f'{won(t["treatmentPrice"])} and takes {t["treatmentDuration"]} minutes. '
                f'{t["treatmentPriceNote"] or ""}\n\n{t["treatmentDesc"]}\n\n'
                f'Performed by: {doctors_of}.'
            )
            if t.get('treatmentAftercare'):
                body += f'\n\n## Aftercare\n\n{t["treatmentAftercare"]}'
            docs.append(
                {
                    'doc_id': f'treatment-{t["_id"]}',
                    'title': f'{t["treatmentName"]} at {name}',
                    'source_type': 'treatment',
                    'lang': 'EN',
                    'text': body,
                    'meta': {
                        'clinicId': c['_id'],
                        'clinicName': name,
                        'treatmentName': t['treatmentName'],
                        'price': t['treatmentPrice'],
                        'duration': t['treatmentDuration'],
                    },
                }
            )
    return docs


def article_documents(api: str) -> list[dict]:
    i = {'page': 1, 'limit': 50, 'search': {'articleCategory': 'HEALTH_TIP'}}
    articles = gql(api, ARTICLES_Q, {'i': i})['getArticles']['list']
    return [
        {
            'doc_id': f'tip-{a["_id"]}',
            'title': a['articleTitle'],
            'source_type': 'health_tip',
            'lang': a.get('articleLang') or 'EN',
            'text': f'# {a["articleTitle"]}\n\n{a["articleContent"]}\n\nWritten by {a["authorName"]}.',
            'meta': {},
        }
        for a in articles
        if len(a['articleContent'].strip()) >= MIN_CONTENT_CHARS
    ]


def notice_documents(api: str) -> list[dict]:
    docs = []
    for category in ['FAQ', 'NOTICE']:
        i = {'page': 1, 'limit': 50, 'noticeCategory': category}
        for n in gql(api, NOTICES_Q, {'i': i})['getNotices']['list']:
            if len(n['noticeContent'].strip()) < MIN_CONTENT_CHARS:
                continue
            docs.append(
                {
                    'doc_id': f'{category.lower()}-{n["_id"]}',
                    'title': n['noticeTitle'],
                    'source_type': category.lower(),
                    'lang': 'EN',
                    'text': f'# {n["noticeTitle"]}\n\n{n["noticeContent"]}',
                    'meta': {'faqCategory': n.get('noticeFaqCategory')},
                }
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://localhost:3000/graphql')
    parser.add_argument('--out', default='data/corpus/medvia.jsonl')
    args = parser.parse_args()
    docs = clinic_documents(args.api) + article_documents(args.api) + notice_documents(args.api)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    counts: dict[str, int] = {}
    for d in docs:
        counts[d['source_type']] = counts.get(d['source_type'], 0) + 1
    print(f'{len(docs)} documents → {out}: {counts}')


if __name__ == '__main__':
    main()
