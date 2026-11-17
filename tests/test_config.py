import pytest
from pydantic import ValidationError

from navi.core.config import Settings


def test_short_api_key_is_refused_at_boot() -> None:
    with pytest.raises(ValidationError, match='at least 32 characters'):
        Settings(api_key='short', _env_file=None)


def test_unknown_environment_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings(api_key='k' * 32, environment='prod', _env_file=None)


def test_key_never_shows_in_repr() -> None:
    s = Settings(api_key='s' * 40, _env_file=None)
    assert 's' * 40 not in repr(s)


def test_collection_with_wrong_dimension_stops_the_app() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    from navi.services.vector_store import VectorStore

    client = QdrantClient(location=':memory:')
    client.create_collection('medvia_knowledge', VectorParams(size=768, distance=Distance.COSINE))
    store = VectorStore(Settings(api_key='k' * 32, _env_file=None), client=client)
    with pytest.raises(RuntimeError, match='size 768'):
        store.ensure_collection()


def test_unknown_dtype_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings(api_key='k' * 32, dtype='int4', _env_file=None)
