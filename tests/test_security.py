import pytest
from fastapi.testclient import TestClient

from tests.conftest import TEST_KEY


def test_ping_with_the_key(client: TestClient) -> None:
    res = client.get('/v1/ping', headers={'X-API-Key': TEST_KEY})
    assert res.status_code == 200
    assert res.json()['service'] == 'medvia-navi'


@pytest.mark.parametrize('headers', [{}, {'X-API-Key': 'wrong'}, {'X-API-Key': TEST_KEY + 'x'}])
def test_ping_without_a_valid_key_is_401(client: TestClient, headers: dict[str, str]) -> None:
    res = client.get('/v1/ping', headers=headers)
    assert res.status_code == 401
    assert res.headers['www-authenticate'] == 'ApiKey'
