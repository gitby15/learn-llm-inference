import asyncio
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """Own the lifecycle and failure boundary of a long-running worker."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._run(),
            name=type(self).__name__,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.run_forever()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception(
                    "%s execution failed; restarting worker",
                    type(self).__name__,
                )
                try:
                    await self.handle_error(error)
                except Exception:
                    logger.exception(
                        "%s failed to report request errors",
                        type(self).__name__,
                    )
                await asyncio.sleep(0)

    async def handle_error(self, error: Exception) -> None:
        """Report a worker-level failure before the processing loop restarts."""

    @abstractmethod
    async def run_forever(self) -> None:
        """Process queued work until cancelled or an unexpected error occurs."""
