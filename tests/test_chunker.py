from navi.domain import Document
from navi.services.chunker import chunk_document

DOC = Document(
    doc_id='clinic-1',
    title='Seoul Glow Dermatology',
    source_type='clinic',
    text=(
        '# Seoul Glow Dermatology\n\nA laser clinic in Gangnam.\n\n'
        '## Opening hours\n\n- Monday: 10:00–19:00\n\n'
        '## Treatments\n\n- Laser Toning: from 99,000 KRW, 30 minutes.'
    ),
)


def test_splits_by_headings_and_keeps_the_heading_path() -> None:
    chunks = chunk_document(DOC)
    assert [c.heading for c in chunks] == [
        'Seoul Glow Dermatology',
        'Seoul Glow Dermatology › Opening hours',
        'Seoul Glow Dermatology › Treatments',
    ]
    assert chunks[2].text == '- Laser Toning: from 99,000 KRW, 30 minutes.'
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_a_long_section_is_split_with_overlap() -> None:
    long = Document(doc_id='g', title='G', source_type='guide', text='# G\n\n' + 'word ' * 500)
    chunks = chunk_document(long, max_chars=900, overlap=120)
    # 2,499 characters, a step of 900 − 120 = 780: pieces start at 0, 780, 1560, 2340.
    assert len(chunks) == 4
    assert all(len(c.text) <= 900 for c in chunks)
    assert chunks[0].text[-120:] == chunks[1].text[:120]


def test_paragraphs_stay_whole_when_they_fit() -> None:
    doc = Document(doc_id='g', title='G', source_type='guide', text='# G\n\nOne.\n\nTwo.')
    assert [c.text for c in chunk_document(doc)] == ['One.\n\nTwo.']


def test_an_h1_that_extends_the_title_is_not_repeated() -> None:
    doc = Document(
        doc_id='c',
        title='Seoul Glow',
        source_type='clinic',
        text='# Seoul Glow (서울글로우)\n\nIntro.\n\n## Hours\n\nMon.',
    )
    assert [c.heading for c in chunk_document(doc)] == ['Seoul Glow', 'Seoul Glow › Hours']


def test_the_chunker_version_is_part_of_the_hash(monkeypatch) -> None:
    from navi.services import ingest

    before = ingest.document_hash(DOC)
    monkeypatch.setattr(ingest, 'CHUNKER_VERSION', 'md-headings-v2')
    assert ingest.document_hash(DOC) != before
