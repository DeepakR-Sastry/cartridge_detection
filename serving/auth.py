"""
Single shared password, HMAC-signed cookie, no dependencies.

The cookie holds an expiry timestamp and a signature over it, keyed by the
password. Readable by anyone, forgeable by nobody without the password.
Changing CF_PASSWORD invalidates every existing session, which is the point.

The check runs as middleware rather than a route dependency because the
StaticFiles mount at "/" is not a route and dependencies do not reach it.
"""

import asyncio
import hashlib
import hmac
import os
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "cf_session"
TTL = int(os.environ.get("CF_SESSION_DAYS", "30")) * 86400
SECURE = os.environ.get("CF_COOKIE_SECURE", "1") != "0"

# /health stays open so the box can be checked without logging in. It reveals
# only which recognizer is loaded. /login obviously has to be open.
OPEN = {"/login", "/health"}

PASSWORD = os.environ.get("CF_PASSWORD", "")
_KEY = hashlib.sha256(f"cartridge-finder:{PASSWORD}".encode()).digest()

router = APIRouter()

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cartridge Finder</title>
<style>
  body { margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#14161a; color:#e8eaed;
         font:16px/1.4 -apple-system, system-ui, sans-serif; }
  form { width:min(340px, 88vw); }
  h1 { font-size:20px; margin:0 0 20px; }
  input, button { font:inherit; width:100%; padding:14px 16px;
                  border-radius:10px; border:1px solid #313640; }
  input { background:#1e2127; color:#e8eaed; margin-bottom:10px; }
  input:focus { outline:none; border-color:#00c853; }
  button { background:#00c853; border-color:#00c853; color:#06140a;
           font-weight:600; }
  p { color:#ff5252; font-size:14px; margin:12px 0 0; }
</style></head><body>
<form method="post" action="/login">
  <h1>Cartridge Finder</h1>
  <input type="password" name="password" autocomplete="current-password"
         placeholder="Password" autofocus>
  <button type="submit">Sign in</button>
  __ERROR__
</form></body></html>"""


def _sign(expiry: int) -> str:
    return hmac.new(_KEY, str(expiry).encode(), hashlib.sha256).hexdigest()[:32]


def _issue() -> str:
    expiry = int(time.time()) + TTL
    return f"{expiry}.{_sign(expiry)}"


def _valid(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    raw, sig = token.rsplit(".", 1)
    if not raw.isdigit():
        return False
    if not hmac.compare_digest(sig, _sign(int(raw))):
        return False
    return int(raw) > time.time()


def _page(error: str = "") -> HTMLResponse:
    body = PAGE.replace("__ERROR__", f"<p>{error}</p>" if error else "")
    return HTMLResponse(body, status_code=401 if error else 200)


@router.get("/login")
def login_page():
    return _page()


@router.post("/login")
async def login(password: str = Form("")):
    if not hmac.compare_digest(password, PASSWORD):
        # Slows brute force over the tunnel to a crawl. Use a long password
        # anyway; this is a speed bump, not a lock.
        await asyncio.sleep(1.0)
        return _page("Wrong password.")

    res = RedirectResponse("/", status_code=303)
    res.set_cookie(
        COOKIE, _issue(),
        max_age=TTL, httponly=True, samesite="lax", secure=SECURE, path="/",
    )
    return res


@router.get("/logout")
def logout():
    res = RedirectResponse("/login", status_code=303)
    res.delete_cookie(COOKIE, path="/")
    return res


def install(app) -> None:
    """Add the guard and the login routes. Call before mounting StaticFiles."""
    if not PASSWORD:
        raise RuntimeError("CF_PASSWORD is not set. Refusing to start open.")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path in OPEN or _valid(request.cookies.get(COOKIE)):
            return await call_next(request)
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/login", status_code=303)
        return JSONResponse({"detail": "not authenticated"}, status_code=401)

    app.include_router(router)