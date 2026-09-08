import asyncio
import inspect
import logging
import time
from functools import wraps
from typing import Any, Protocol, TypeVar


logger = logging.getLogger(__name__)


class Worker(Protocol):
    def start(self) -> None: ...

    async def stop(self) -> None: ...


WorkerType = TypeVar("WorkerType")


def worker_lifecycle(worker_class: type[WorkerType]) -> type[WorkerType]:
    """Add a managed asyncio task lifecycle to a worker class."""
    if not inspect.iscoroutinefunction(
        getattr(worker_class, "_process_task", None)
    ):
        raise TypeError(
            f"{worker_class.__name__} must define an async _process_task method"
        )

    original_init = worker_class.__init__

    @wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._task = None

    def start(self: Any) -> None:
        task = self._task
        if task is not None and not task.done():
            return

        timestamp_suffix = f"{time.time()}"
        task_name = f"{type(self).__name__}-{timestamp_suffix}"

        async def run_process_task() -> None:
            try:
                await self._process_task()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s execution failed", type(self).__name__)
                raise

        self._task = asyncio.create_task(run_process_task(), name=task_name)

    async def stop(self: Any) -> None:
        task = self._task
        if task is None:
            return

        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task = None

    setattr(worker_class, "__init__", __init__)
    setattr(worker_class, "start", start)
    setattr(worker_class, "stop", stop)
    return worker_class
