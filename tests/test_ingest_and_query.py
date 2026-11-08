from fastapi.testclient import TestClient

from tests.conftest import TEST_KEY

H = {'X-API-Key': TEST_KEY}


def doc(doc_id: str, text: str, title: str = 'Seoul Glow Dermatology') -> dict:
    return {'doc_id': doc_id, 'title': title, 'source_type': 'clinic', 'text': text}


CLINIC = doc(
    'clinic-1',
    '# Seoul Glow Dermatology\n\nA laser clinic in Gangnam.\n\n'
    '## Treatments\n\nLaser Toning costs from 99,000 KRW and takes 30 minutes.',
)


def test_ingest_then_unchanged_then_replaced(client: TestClient) -> None:
    first = client.post('/v1/documents', json={'documents': [CLINIC]}, headers=H).json()
    assert first['items'] == [{'doc_id': 'clinic-1', 'status': 'indexed', 'chunks': 2}]

    again = client.post('/v1/documents', json={'documents': [CLINIC]}, headers=H).json()
    assert again['items'][0]['status'] == 'unchanged'

    shorter = doc('clinic-1', '# Seoul Glow Dermatology\n\nMoved to Seocho.')
    client.post('/v1/documents', json={'documents': [shorter]}, headers=H)
    # The old second chunk is gone, not left behind.
    assert client.get('/v1/documents/clinic-1', headers=H).json()['chunks'] == 1


def test_delete_removes_every_chunk(client: TestClient) -> None:
    client.post('/v1/documents', json={'documents': [CLINIC]}, headers=H)
    assert client.delete('/v1/documents/clinic-1', headers=H).json()['chunks'] == 2
    assert client.get('/v1/documents/clinic-1', headers=H).status_code == 404


def test_ingest_refuses_bad_input(client: TestClient) -> None:
    bad_id = doc('../etc/passwd', 'x')
    assert client.post('/v1/documents', json={'documents': [bad_id]}, headers=H).status_code == 422
    dup = {'documents': [CLINIC, CLINIC]}
    assert client.post('/v1/documents', json=dup, headers=H).status_code == 422
    assert client.post('/v1/documents', json={'documents': [CLINIC]}).status_code == 401


def test_query_answers_with_a_cited_source(client: TestClient) -> None:
    client.post('/v1/documents', json={'documents': [CLINIC]}, headers=H)
    res = client.post('/v1/query', json={'question': 'How much does Laser Toning cost?'}, headers=H)
    body = res.json()
    assert res.status_code == 200
    assert body['refused'] is False
    assert body['used_llm'] is True
    assert body['sources'][0]['cited'] is True
    assert '99,000' in body['sources'][0]['text']


def test_query_with_nothing_relevant_refuses_without_the_model(client: TestClient) -> None:
    client.post('/v1/documents', json={'documents': [CLINIC]}, headers=H)
    res = client.post('/v1/query', json={'question': 'Погода в Пусане завтра?'}, headers=H)
    body = res.json()
    assert body == {
        **body,
        'refused': True,
        'used_llm': False,
        'sources': [],
        'answer': 'В материалах MedVia нет такой информации.',
    }


def test_query_validates_length(client: TestClient) -> None:
    assert client.post('/v1/query', json={'question': 'x' * 501}, headers=H).status_code == 422
