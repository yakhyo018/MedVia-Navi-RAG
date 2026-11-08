from fastapi.testclient import TestClient

from navi.core.config import Settings
from navi.main import create_app


def test_live_needs_no_key(client: TestClient) -> None:
    res = client.get('/health/live')
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'


def test_ready_reports_qdrant(client: TestClient) -> None:
    res = client.get('/health/ready')
    assert res.status_code == 200
    assert res.json()['qdrant'] == 'ok'


def test_ready_is_503_when_qdrant_is_down(client: TestClient) -> None:
    client.app.state.store.is_ready = lambda: False  # type: ignore[attr-defined]
    res = client.get('/health/ready')
    assert res.status_code == 503
    assert res.json() == {'status': 'unavailable', 'version': '0.1.0', 'qdrant': 'down'}


def test_docs_are_hidden_in_production(settings: Settings) -> None:

    prod = settings.model_copy(update={'environment': 'production'})
    with TestClient(create_app(prod)) as c:
        assert c.get('/docs').status_code == 404
        assert c.get('/openapi.json').status_code == 404
