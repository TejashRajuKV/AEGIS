"""FastAPI Dependency Injection Functions."""
from typing import AsyncGenerator
from .models.database import get_session
from .services.cache import InMemoryCache as Cache
from .services.task_queue import TaskQueue
from .services.websocket_manager import WebSocketManager

_cache_instance: Cache | None = None
_task_queue_instance: TaskQueue | None = None
_ws_manager_instance: WebSocketManager | None = None


async def get_db():
    """Yield an async SQLAlchemy database session (AsyncSession)."""
    async for session in get_session():
        yield session


def get_cache() -> Cache:
    """Return a singleton Cache instance."""
    global _cache_instance
    if _cache_instance is None:
        from .config import settings
        _cache_instance = Cache(max_size=settings.cache_max_size, ttl=settings.cache_ttl)
    return _cache_instance


def get_task_queue() -> TaskQueue:
    """Return a singleton TaskQueue instance."""
    global _task_queue_instance
    if _task_queue_instance is None:
        _task_queue_instance = TaskQueue()
    return _task_queue_instance


def get_websocket_manager() -> WebSocketManager:
    """Return a singleton WebSocketManager instance."""
    global _ws_manager_instance
    if _ws_manager_instance is None:
        _ws_manager_instance = WebSocketManager()
    return _ws_manager_instance
