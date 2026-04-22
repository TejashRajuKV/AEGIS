"""
TaskQueue – background task queue with sequential execution.

Designed for 16 GB RAM: only ONE task runs at a time.  Pending tasks
wait in a priority queue until the current task finishes.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class TaskItem:
    """A queued task with priority ordering."""

    priority: int
    task_id: str = field(compare=False)
    func: Callable[..., Any] = field(compare=False)
    args: Tuple = field(compare=False, default=())
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    created_at: float = field(compare=False, default_factory=time.time)
    started_at: Optional[float] = field(compare=False, default=None)
    completed_at: Optional[float] = field(compare=False, default=None)


class TaskQueue:
    """Async background task queue with **strict sequential execution**.

    Only one task is processed at a time.  New tasks are enqueued and
    processed in priority order (lower number = higher priority).

    Parameters
    ----------
    max_concurrent:
        Maximum tasks running simultaneously.  **Must be 1** for the
        16 GB RAM constraint.  Kept as a parameter for future flexibility.
    retention_seconds:
        How long to keep completed/failed task metadata before cleanup.
    default_priority:
        Default priority for enqueued tasks (lower = higher priority).
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        retention_seconds: float = 3600.0,
        default_priority: int = 10,
    ) -> None:
        self.max_concurrent = max(1, max_concurrent)  # enforce minimum 1
        self.retention_seconds = retention_seconds
        self.default_priority = default_priority

        self._queue: List[TaskItem] = []
        self._tasks: Dict[str, TaskItem] = {}
        self._running_count: int = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._processing = False
        self._total_processed: int = 0

        logger.info(
            "TaskQueue initialised (max_concurrent=%d, retention=%.0fs)",
            self.max_concurrent,
            self.retention_seconds,
        )

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------
    async def enqueue(
        self,
        task_func: Callable[..., Any],
        *args: Any,
        priority: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Add a task to the queue.

        Parameters
        ----------
        task_func:
            Callable to execute.  Can be sync or async.
        *args, **kwargs:
            Positional and keyword arguments for *task_func*.
        priority:
            Task priority (lower = higher priority).  If None, uses
            the default priority.

        Returns
        -------
        task_id : str
        """
        async with self._lock:
            task_id = str(uuid.uuid4())[:12]
            pri = priority if priority is not None else self.default_priority

            item = TaskItem(
                priority=pri,
                task_id=task_id,
                func=task_func,
                args=args,
                kwargs=kwargs,
            )
            self._queue.append(item)
            self._tasks[task_id] = item

            # Keep queue sorted by priority
            self._queue.sort(key=lambda t: t.priority)

            logger.info(
                "Task enqueued: id=%s, priority=%d, queue_size=%d",
                task_id,
                pri,
                len(self._queue),
            )

        # Kick off processing if idle — Bug 21 fix: check and set _processing
        # atomically under the lock so two concurrent enqueue() calls can't
        # both spawn a _process_loop() coroutine.
        async with self._lock:
            if not self._processing:
                self._processing = True  # claim the loop slot before creating task
                asyncio.create_task(self._process_loop())

        return task_id

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------
    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return status and metadata for a task."""
        item = self._tasks.get(task_id)
        if item is None:
            return None

        elapsed = 0.0
        if item.started_at:
            end = item.completed_at or time.time()
            elapsed = end - item.started_at

        return {
            "task_id": item.task_id,
            "status": item.status.value,
            "priority": item.priority,
            "created_at": item.created_at,
            "started_at": item.started_at,
            "completed_at": item.completed_at,
            "elapsed_seconds": round(elapsed, 3),
            "error": item.error,
            "has_result": item.result is not None,
        }

    def get_result(self, task_id: str) -> Optional[Any]:
        """Return the result of a completed task, or None."""
        item = self._tasks.get(task_id)
        if item is None:
            return None
        if item.status == TaskStatus.COMPLETED:
            return item.result
        return None

    def get_queue_size(self) -> int:
        """Return the number of pending tasks."""
        return sum(1 for t in self._queue if t.status == TaskStatus.PENDING)

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Return status for all tasks."""
        return [self.get_status(tid) for tid in self._tasks if self.get_status(tid)]

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------
    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending task.  Cannot cancel running tasks.

        Returns True if the task was successfully cancelled.
        """
        async with self._lock:
            item = self._tasks.get(task_id)
            if item is None:
                return False
            if item.status == TaskStatus.RUNNING:
                logger.warning("Cannot cancel running task %s", task_id)
                return False
            if item.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False

            item.status = TaskStatus.CANCELLED
            self._queue = [t for t in self._queue if t.task_id != task_id]
            logger.info("Task %s cancelled", task_id)
            return True

    # ------------------------------------------------------------------
    # Processing loop (sequential)
    # ------------------------------------------------------------------
    async def _process_loop(self) -> None:
        """Main processing loop – processes tasks one at a time.

        Bug 21 fix: _processing is now set to True by enqueue() before this
        coroutine is created, so there is no window where two loops can start.
        """
        logger.debug("TaskQueue processing loop started")

        try:
            while True:
                # Bug 22 fix: select and mark item RUNNING inside a single
                # lock acquisition so cancel() cannot sneak in between.
                async with self._lock:
                    pending = [t for t in self._queue if t.status == TaskStatus.PENDING]
                    if not pending:
                        break
                    item = pending[0]  # already sorted by priority
                    item.status = TaskStatus.RUNNING
                    item.started_at = time.time()

                await self._process_one(item)
                self._total_processed += 1

                # Periodic cleanup
                if self._total_processed % 20 == 0:
                    self._cleanup()
        finally:
            self._processing = False
            logger.debug("TaskQueue processing loop stopped")

    async def _process_one(self, item: TaskItem) -> None:
        """Process a single task (with semaphore for concurrency control).

        Bug 22 fix: item is already marked RUNNING by _process_loop() before
        this method is called, so we skip the redundant status update here.

        Fix CRIT-04: sync callables are executed via run_in_executor so they
        never block the asyncio event loop (important for LLM retry loops that
        call time.sleep). Async callables are awaited directly.
        """
        async with self._semaphore:
            logger.info("Task %s starting (priority=%d)", item.task_id, item.priority)

            try:
                if asyncio.iscoroutinefunction(item.func):
                    # Async callable — await directly
                    result = await item.func(*item.args, **item.kwargs)
                else:
                    # Fix CRIT-04: sync callable — run in thread pool to keep
                    # the event loop free during blocking operations (e.g., LLM
                    # API calls with time.sleep retry backoff).
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: item.func(*item.args, **item.kwargs),
                    )

                async with self._lock:
                    item.result = result
                    item.status = TaskStatus.COMPLETED
                    item.completed_at = time.time()
                    # Fix CRIT-08: queue mutation under lock
                    self._queue = [t for t in self._queue if t.task_id != item.task_id]

                logger.info(
                    "Task %s completed in %.3fs",
                    item.task_id,
                    item.completed_at - (item.started_at or item.completed_at),
                )

            except asyncio.CancelledError:
                async with self._lock:
                    item.status = TaskStatus.CANCELLED
                    item.error = "Task was cancelled"
                    item.completed_at = time.time()
                    self._queue = [t for t in self._queue if t.task_id != item.task_id]
                logger.info("Task %s cancelled", item.task_id)

            except Exception as exc:
                async with self._lock:
                    item.status = TaskStatus.FAILED
                    item.error = str(exc)
                    item.completed_at = time.time()
                    self._queue = [t for t in self._queue if t.task_id != item.task_id]
                logger.error("Task %s failed: %s", item.task_id, exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        """Remove completed/failed tasks older than retention period."""
        now = time.time()
        to_remove = []
        for tid, item in self._tasks.items():
            if item.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if item.completed_at and (now - item.completed_at) > self.retention_seconds:
                    to_remove.append(tid)

        for tid in to_remove:
            del self._tasks[tid]

        if to_remove:
            logger.debug("Cleaned up %d expired tasks", len(to_remove))

    async def clear_completed(self) -> int:
        """Manually clear all completed/failed/cancelled tasks."""
        async with self._lock:
            to_remove = [
                tid for tid, item in self._tasks.items()
                if item.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            for tid in to_remove:
                del self._tasks[tid]
            self._queue = [t for t in self._queue if t.status == TaskStatus.PENDING]
            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        status_counts = {s.value: 0 for s in TaskStatus}
        for item in self._tasks.values():
            status_counts[item.status.value] += 1

        return {
            "total_tasks": len(self._tasks),
            "queue_size": self.get_queue_size(),
            "max_concurrent": self.max_concurrent,
            "total_processed": self._total_processed,
            "is_processing": self._processing,
            "by_status": status_counts,
        }
