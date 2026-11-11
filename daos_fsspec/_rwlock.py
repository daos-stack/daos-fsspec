# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Writer-preferring shared/exclusive mount lock. Native calls share access; teardown and reconnect require exclusive access."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType

#: Default deferral window (seconds); see the module docstring.
DEFAULT_READER_DEFER_TIMEOUT = 5.0


class RWLock:
    """A reentrant, writer-preferring shared/exclusive lock."""

    __slots__ = (
        "_cond",
        "_readers",
        "_writer",
        "_writer_depth",
        "_waiting_writers",
        "_local",
        "_read_guard",
        "reader_defer_timeout",
    )

    def __init__(
        self,
        reader_defer_timeout: float = DEFAULT_READER_DEFER_TIMEOUT,
        *,
        read_guard: Callable[[], None] | None = None,
    ) -> None:
        #: How long a newly-arriving reader defers to a *waiting* writer
        #: before proceeding anyway. Bounds the damage if a caller ever holds
        #: the shared lock while waiting on threads that need it themselves.
        self.reader_defer_timeout = reader_defer_timeout
        self._cond = threading.Condition(threading.Lock())
        # Total active shared acquisitions across all threads, NOT counting
        # reentrant nesting. Teardown waits for this to reach zero.
        self._readers = 0
        # Thread ident holding the exclusive lock, or None.
        self._writer: int | None = None
        self._writer_depth = 0
        # Writers parked in acquire_write(). Newly-arriving readers defer to
        # these (bounded by reader_defer_timeout) so teardown can drain.
        self._waiting_writers = 0
        self._local = threading.local()
        # Optional owner-supplied admission check, run *after* the shared lock
        # is acquired. DFSFileSystem uses this to make its closed/fork check
        # atomic with operation admission: checking before acquire leaves a
        # window where close() can tear down the mount while a reader waits.
        self._read_guard = read_guard

    # ── shared (read) ────────────────────────────────────────────────────

    def acquire_read(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Take the shared lock. Returns False if it could not be taken."""
        depth = getattr(self._local, "rdepth", 0)
        if depth:
            # Already held by this thread: nest without touching the
            # condition, so a waiting writer can never deadlock us.
            self._local.rdepth = depth + 1
            return True

        me = threading.get_ident()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            # A thread holding the exclusive lock already has stronger access.
            if self._writer != me:
                defer_until = time.monotonic() + self.reader_defer_timeout
                while self._writer is not None or (
                    self._waiting_writers > 0 and time.monotonic() < defer_until
                ):
                    if not blocking:
                        return False
                    now = time.monotonic()
                    wait_for: float | None = None
                    if deadline is not None:
                        wait_for = deadline - now
                        if wait_for <= 0:
                            return False
                    # Wake when the deferral window closes so we can re-check
                    # rather than sleeping past it waiting for a notify.
                    if self._writer is None and self._waiting_writers > 0:
                        remaining_defer = defer_until - now
                        wait_for = (
                            remaining_defer if wait_for is None else min(wait_for, remaining_defer)
                        )
                    self._cond.wait(wait_for)
            self._readers += 1
        self._local.rdepth = 1
        return True

    def release_read(self) -> None:
        depth = getattr(self._local, "rdepth", 0)
        if depth <= 0:
            raise RuntimeError("release_read() called without holding the shared lock")
        if depth > 1:
            self._local.rdepth = depth - 1
            return
        self._local.rdepth = 0
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                # Teardown may be waiting for the last reader to leave.
                self._cond.notify_all()

    @contextmanager
    def read(self) -> Iterator[None]:
        """Context manager for the shared lock (normal I/O path)."""
        self.acquire_read()
        try:
            if self._read_guard is not None:
                self._read_guard()
            yield
        finally:
            self.release_read()

    @contextmanager
    def _read_unchecked(self) -> Iterator[None]:
        """Take the shared lock without running the owner's admission guard.

        This is for generation-aware cleanup only. Normal I/O must use
        :meth:`read`, otherwise it can enter after the owner has closed.
        """
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    # ── exclusive (write) ────────────────────────────────────────────────

    def acquire_write(self, timeout: float | None = None) -> bool:
        """Take the exclusive lock, waiting for in-flight readers to drain."""
        me = threading.get_ident()
        if self._writer == me:
            self._writer_depth += 1
            return True
        if getattr(self._local, "rdepth", 0):
            # Upgrading would wait for a reader count that includes ourselves.
            raise RuntimeError(
                "cannot acquire the exclusive lock while holding the shared lock "
                "(lock upgrade would deadlock); release the shared lock first"
            )

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            # Announce intent before waiting: newly-arriving readers defer to
            # us, so the reader count can actually reach zero under load.
            self._waiting_writers += 1
            try:
                while self._writer is not None or self._readers > 0:
                    if deadline is None:
                        self._cond.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not self._cond.wait(remaining):
                            if self._writer is not None or self._readers > 0:
                                return False
                self._writer = me
                self._writer_depth = 1
            finally:
                self._waiting_writers -= 1
                if self._waiting_writers == 0:
                    # Release any readers that were deferring to us: matters
                    # on the give-up path, where nobody else will wake them.
                    self._cond.notify_all()
        return True

    def release_write(self) -> None:
        me = threading.get_ident()
        if self._writer != me:
            raise RuntimeError("release_write() called without holding the exclusive lock")
        if self._writer_depth > 1:
            self._writer_depth -= 1
            return
        with self._cond:
            self._writer = None
            self._writer_depth = 0
            self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        """Context manager for the exclusive lock (teardown / reconnect)."""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    # ── convenience ──────────────────────────────────────────────────────

    def __enter__(self) -> RWLock:
        # Defaults to the shared lock: the overwhelmingly common case, and the
        # safe one: a caller who wanted exclusivity must say so explicitly.
        self.acquire_read()
        if self._read_guard is not None:
            try:
                self._read_guard()
            except BaseException:
                self.release_read()
                raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release_read()

    @property
    def readers(self) -> int:
        """Current shared-lock holders. For tests and diagnostics."""
        with self._cond:
            return self._readers
