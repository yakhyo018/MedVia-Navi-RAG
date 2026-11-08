from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from navi import __version__

router = APIRouter(prefix='/health', tags=['health'])


class HealthStatus(BaseModel):
    status: str
    version: str
    qdrant: str | None = None


@router.get('/live', response_model=HealthStatus)
def live() -> HealthStatus:
    """
    Liveness: the process runs. Checks nothing else on purpose — if this depended on Qdrant,
    an orchestrator (Docker, Kubernetes) would restart a healthy API every time Qdrant blinks.
    """
    return HealthStatus(status='ok', version=__version__)


@router.get('/ready', response_model=HealthStatus)
def ready(request: Request, response: Response) -> HealthStatus:
    """Readiness: may take traffic — Qdrant answers. 503 otherwise, so a load balancer skips it."""
    if request.app.state.store.is_ready():
        return HealthStatus(status='ok', version=__version__, qdrant='ok')
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthStatus(status='unavailable', version=__version__, qdrant='down')
