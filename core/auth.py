"""Authentication, borrowed from veeragenai_projects_be.

No project in this service issues tokens. Each verifies the same HS256 access
token that the platform's auth backend sets as a cookie, using the shared
JWT_SECRET, so a user signed in to the workspace is signed in here too with the
same user id. Every project scopes its queries by that id.
"""

import jwt
from fastapi import Cookie, Header, HTTPException, Request, status

from core.config import settings


def decode_access_token(access_token: str | None) -> str | None:
    if not access_token:
        return None
    try:
        return jwt.decode(access_token, settings.jwt_secret, algorithms=["HS256"]).get("sub")
    except (jwt.InvalidTokenError, KeyError):
        return None


def _extract_token(request: Request, access_token: str | None, authorization: str | None) -> str | None:
    token = access_token
    if not token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        elif len(parts) == 1:
            token = parts[0]
    return token or request.cookies.get("access_token")


async def current_user_id(
    request: Request,
    access_token: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    user_id = decode_access_token(_extract_token(request, access_token, authorization))
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user_id
