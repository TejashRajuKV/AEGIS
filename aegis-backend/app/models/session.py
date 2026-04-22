"""
AEGIS AI Fairness Platform - Database Session Dependency.

Provides an async generator that yields SQLAlchemy sessions and
ensures they are properly closed after each request.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    The session is automatically closed when the request completes.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
