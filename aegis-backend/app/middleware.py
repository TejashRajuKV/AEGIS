"""AEGIS Middleware - CORS, request-ID tracing, timing."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestMetaMiddleware(BaseHTTPMiddleware):
    """
    Adds to every response:
    - ``X-Request-ID``:   unique UUID per request for distributed tracing
    - ``X-Process-Time``: accurate wall-clock time via perf_counter (ms)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()                              # perf_counter > time.time()

        response: Response = await call_next(request)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms}ms"

        from app.utils.logger import get_logger
        logger = get_logger("aegis.middleware")
        logger.debug(
            "%s %s → %d  [%sms] [req:%s]",
            request.method, request.url.path,
            response.status_code, elapsed_ms, request_id,
        )
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with timing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        method = request.method
        path = request.url.path

        response = await call_next(request)

        duration_ms = (time.time() - start) * 1000
        status = response.status_code

        from app.utils.logger import get_logger
        logger = get_logger("aegis.middleware")
        logger.info(f"{method} {path} -> {status} ({duration_ms:.1f}ms)")

        response.headers["X-Process-Time"] = f"{duration_ms:.2f}"
        return response


def setup_middleware(app):
    """Configure all middleware for the application."""
    # CORS is added directly in main.py
    # RequestMetaMiddleware: X-Request-ID + accurate perf_counter timing
    app.add_middleware(RequestMetaMiddleware)
    # RequestLoggingMiddleware: INFO-level access log per request
    app.add_middleware(RequestLoggingMiddleware)
