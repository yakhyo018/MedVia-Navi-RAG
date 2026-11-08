from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from navi.core.config import Settings, get_settings
from navi.main import create_app

TEST_KEY = 'test-key-' + 'x' * 32


@pytest.fixture
def settings() -> Settings:
    """In-memory Qdrant, a fixed key: no Docker, no .env, the same on every machine and in CI."""
    return Settings(
        api_key=TEST_KEY,
        environment='test',
        qdrant_url=':memory:',
        embedder='hash',
        generator='fake',
        min_score=0.1,
        _env_file=None,  # never read a developer's local .env inside the tests
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    # require_api_key reads Settings through Depends(get_settings): point it at the test ones.
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:  # `with` runs the lifespan: startup + shutdown
        yield c
