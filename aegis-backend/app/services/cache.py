"""
InMemoryCache – thread-safe LRU cache with optional TTL.

Designed for caching API responses, model predictions, and intermediate
computation results within the AEGIS backend.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with value, TTL, and access metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    ttl_seconds: Optional[float] = None
    access_count: int = 0


class InMemoryCache:
    """Thread-safe in-memory LRU cache with optional TTL support.

    Parameters
    ----------
    max_size:
        Maximum number of entries to store.  When exceeded, the least
        recently used entry is evicted.
    default_ttl:
        Default time-to-live in seconds for entries.  None means no
        expiry (entries live until evicted by LRU policy).
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        logger.info(
            "InMemoryCache initialised (max_size=%d, default_ttl=%s)",
            max_size,
            default_ttl,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache.

        Returns None if the key doesn't exist or has expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check expiry
            if self._is_expired(entry):
                self._remove_entry(key)
                self._misses += 1
                logger.debug("Cache miss (expired): %s", key)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hits += 1

            logger.debug("Cache hit: %s (accessed %d times)", key, entry.access_count)
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store a value in the cache.

        Parameters
        ----------
        key:
            Cache key.
        value:
            Value to store (must be picklable for advanced use cases).
        ttl_seconds:
            Time-to-live in seconds.  Overrides the default TTL.
            None means use the default; 0 means no expiry.
        """
        with self._lock:
            # If key exists, remove old entry first (to update position)
            if key in self._cache:
                del self._cache[key]

            now = time.time()
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl,
            )
            self._cache[key] = entry

            # Evict if over capacity
            while len(self._cache) > self.max_size:
                self._evict_lru()

            logger.debug("Cache set: %s (ttl=%s, size=%d/%d)", key, ttl, len(self._cache), self.max_size)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Returns True if the key existed and was removed.
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                logger.debug("Cache delete: %s", key)
                return True
            return False

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns the number of entries removed.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared (%d entries removed)", count)
            return count

    # ------------------------------------------------------------------
    # Compute-on-miss
    # ------------------------------------------------------------------
    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], T],
        ttl_seconds: Optional[float] = None,
    ) -> T:
        """Get a value from cache, computing it if missing.

        Fix MED-03: guards against the TOCTOU race where two concurrent callers
        both see a cache miss and both execute compute_func() (potentially an
        expensive ML operation).  Uses a per-key in-flight threading.Event so
        only the first caller computes; all others wait and reuse the result.

        Parameters
        ----------
        key:
            Cache key.
        compute_func:
            Zero-argument callable that computes the value.
        ttl_seconds:
            TTL for the cached result.

        Returns
        -------
        The cached or freshly computed value.
        """
        import threading as _threading

        # Fast path: value already cached
        value = self.get(key)
        if value is not None:
            return value

        # Slow path: need to compute.  Check whether another caller is already
        # computing this key, or claim the compute slot ourselves.
        with self._lock:
            # Re-check under lock after acquiring it (double-checked locking)
            entry = self._cache.get(key)
            if entry is not None and not self._is_expired(entry):
                return entry.value  # type: ignore[attr-defined]

            # Check if another thread is already computing this key
            if not hasattr(self, "_inflight"):
                self._inflight: dict = {}

            if key in self._inflight:
                # Another caller is computing – grab the event and wait outside lock
                event: _threading.Event = self._inflight[key]
                is_computing = False
            else:
                # We are the first caller – register our event
                event = _threading.Event()
                self._inflight[key] = event
                is_computing = True

        if not is_computing:
            # Wait for the computing thread to finish, then read from cache
            logger.debug("Cache – waiting for in-flight compute: %s", key)
            event.wait(timeout=300)  # 5-minute safety timeout
            return self.get(key)  # type: ignore[return-value]

        # We are the designated compute thread
        logger.debug("Cache miss – computing: %s", key)
        try:
            result = compute_func()
        except Exception as exc:
            logger.error("Compute function failed for key %s: %s", key, exc)
            with self._lock:
                self._inflight.pop(key, None)
            event.set()
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

        self.set(key, result, ttl_seconds=ttl_seconds)
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def keys(self) -> List[str]:
        """Return all non-expired cache keys."""
        with self._lock:
            self._purge_expired()
            return list(self._cache.keys())

    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                self._remove_entry(key)
                return False
            return True

    def size(self) -> int:
        """Return the number of non-expired entries."""
        with self._lock:
            self._purge_expired()
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            self._purge_expired()
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
                "evictions": self._evictions,
                "total_requests": total_requests,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has exceeded its TTL."""
        if entry.ttl_seconds is None:
            return False
        return (time.time() - entry.created_at) > entry.ttl_seconds

    def _remove_entry(self, key: str) -> None:
        """Remove a single entry by key."""
        self._cache.pop(key, None)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug("LRU eviction: %s", key)

    def _purge_expired(self) -> int:
        """Remove all expired entries.  Must be called with lock held."""
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if entry.ttl_seconds is not None
            and (now - entry.created_at) > entry.ttl_seconds
        ]
        for key in expired:
            self._remove_entry(key)
            self._evictions += 1
        if expired:
            logger.debug("Purged %d expired entries", len(expired))
        return len(expired)
