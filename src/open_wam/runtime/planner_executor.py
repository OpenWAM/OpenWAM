"""Single-owner speculative work: cancel queued work and drain running model calls."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar
from collections.abc import Callable

Result = TypeVar("Result")
Request = TypeVar("Request")


@dataclass(frozen=True)
class PlannerTeardown:
    cancelled: bool = False
    error: str | None = None
    drain_time_s: float = 0.0


class PlannerExecutor(Generic[Result]):
    """Serialize model access; live failures propagate, discarded failures are receipts.

    Python cannot safely interrupt a running model call. close() must finish it
    before the next session may use that model, but never publish its candidate
    or let its exception replace an already terminal environment outcome.
    """

    def __init__(self) -> None:
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._pending: Future[Result] | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def submit(self, fn: Callable[[Request], Result], request: Request) -> None:
        if self.pending:
            raise RuntimeError("Only one planner transaction may be pending.")
        self._pending = self._worker.submit(fn, request)

    def take(self) -> Result:
        if self._pending is None:
            raise RuntimeError("No pending planner transaction.")
        pending, self._pending = self._pending, None
        return pending.result()

    def close(self) -> PlannerTeardown:
        began = time.perf_counter()
        pending, self._pending = self._pending, None
        cancelled, error = False, None
        try:
            if pending is not None:
                cancelled = pending.cancel()
                if not cancelled:
                    try:
                        pending.result()
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
        finally:
            self._worker.shutdown(wait=True, cancel_futures=True)
        return PlannerTeardown(cancelled, error, time.perf_counter() - began)
