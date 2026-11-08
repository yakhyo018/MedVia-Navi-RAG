import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from navi import __version__
from navi.api import documents, health, query
from navi.container import Container, build_container
from navi.core.config import Settings, get_settings
from navi.core.security import require_api_key


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """
    App factory, not a module-level `app`: each test builds its own app with its own settings
    and an in-memory store, and nothing connects to anything at import time.
    """
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: connect, check the collection, load both models — before the first request,
        # so the first patient does not wait 20 seconds for a model to load.
        app.state.container = container or build_container(settings)
        app.state.store = app.state.container.store
        yield
        # Shutdown: close the connection pool.
        app.state.store.close()

    app = FastAPI(
        title='MedVia Navi',
        version=__version__,
        lifespan=lifespan,
        # The interactive docs list every route; not on a production server.
        docs_url=None if settings.environment == 'production' else '/docs',
        redoc_url=None,
        openapi_url=None if settings.environment == 'production' else '/openapi.json',
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    # Everything except /health needs the key.
    @app.get('/v1/ping', dependencies=[Depends(require_api_key)], tags=['meta'])
    def ping() -> dict[str, str]:
        return {'service': settings.app_name, 'version': __version__}

    return app


def run() -> None:
    """`navi-api` entry point for local runs; Docker calls uvicorn directly."""
    import uvicorn

    uvicorn.run('navi.main:create_app', factory=True, host='127.0.0.1', port=8100)  # noqa: S104
