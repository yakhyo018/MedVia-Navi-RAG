import pytest

from navi.domain import RetrievedChunk
from navi.services.prompting import (
    REFUSALS,
    build_messages,
    citations,
    detect_lang,
    is_refusal,
    refusal_for,
)


@pytest.mark.parametrize(
    ('text', 'lang'),
    [
        ('How much is laser toning?', 'EN'),
        ('Сколько стоит лазерное тонирование?', 'RU'),
        ('Lazer toning qancha turadi?', 'UZ'),
        ("Bron qanday bekor qilinadi? O'zbekcha", 'UZ'),
        ('Лазерлік тонинг қанша тұрады?', 'KK'),
    ],
)
def test_detect_lang(text: str, lang: str) -> None:
    assert detect_lang(text) == lang


def test_refusal_follows_the_question_language() -> None:
    assert refusal_for('Где клиника?') == REFUSALS['RU']


def test_messages_number_the_sources_and_name_the_refusal() -> None:
    chunk = RetrievedChunk(
        doc_id='d',
        chunk_index=0,
        title='T',
        heading='T › H',
        text='Body.',
        source_type='faq',
        score=0.9,
    )
    system, user = build_messages('Is it free?', [chunk, chunk])
    assert REFUSALS['EN'] in system['content']
    assert '[1] T › H\nBody.' in user['content']
    assert '[2] T › H' in user['content']
    assert user['content'].endswith('Question: Is it free?')


def test_citations_in_order_without_duplicates() -> None:
    assert citations('From 99,000 KRW [2]. Thirty minutes [1][2].') == [2, 1]


@pytest.mark.parametrize(
    'answer',
    [
        REFUSALS['EN'],
        REFUSALS['RU'],
        'The sources do not say.',
        'This is not mentioned in the sources.',
    ],
)
def test_refusals_are_recognised(answer: str) -> None:
    assert is_refusal(answer)


def test_an_answer_is_not_a_refusal() -> None:
    assert not is_refusal('Laser toning costs from 99,000 KRW [1].')
