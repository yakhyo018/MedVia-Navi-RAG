import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from navi.core.config import Settings, get_settings

# auto_error=False: a missing header reaches require_api_key, which answers 401 itself —
# FastAPI's default would answer 403, and "no credentials" is 401.
_api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


def require_api_key(
    provided: str | None = Depends(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Navi has no users of its own: only MedVia's services call it. One shared key, compared in
    constant time (hmac.compare_digest) — `==` stops at the first different byte, and the time
    it takes tells an attacker how many leading bytes were right (a timing attack).
    """
    expected = settings.api_key.get_secret_value()
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or missing API key',
            headers={'WWW-Authenticate': 'ApiKey'},
        )
