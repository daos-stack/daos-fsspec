# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSFileSystemPool: connection pool for DAOS DFS.

Maintains multiple DFSFileSystem instances for concurrent access.
Each instance holds an independent DFS handle with its own lock,
so operations on different handles don't contend.

Usage::

    from daos_fsspec.pool import DFSFileSystemPool

    pool = DFSFileSystemPool(pool="mypool", container="mycont", size=4)
    fs = pool.acquire()
    try:
        data = fs.cat("/path/to/file")
    finally:
        pool.release(fs)

    # Or use the context manager:
    with pool.connection() as fs:
        data = fs.cat("/path/to/file")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from daos_fsspec import _fork
from daos_fsspec.core import DFSFileSystem

_logger = logging.getLogger(__name__)


class DFSFileSystemPool:
    """Connection pool for DFSFileSystem.

    Maintains a pool of N open DFSFileSystem instances. Each instance
    has its own lock, so operations on different instances don't contend.

    Parameters
    ----------
    pool : str
        DAOS pool label.
    container : str
        DAOS container label.
    size : int
        Number of connections in the pool (default: 4).
    readonly : bool
        Mount read-only (default: False).
    retry_count : int
        Number of retries for transient errors (default: 3).
    retry_backoff : float
        Initial backoff in seconds (default: 0.1).
    timeout : float | None
        Timeout in seconds (default: None).
    """

    def __init__(
        self,
        pool: str,
        container: str,
        size: int = 4,
        readonly: bool = False,
        object_class: int = 0,
        chunk_size: int = 0,
        retry_count: int = 3,
        retry_backoff: float = 0.1,
        **kwargs: Any,
    ):
        _fork._check_can_initialize()
        if size < 1:
            raise ValueError("pool size must be >= 1")

        self._pool_label = pool
        self._cont_label = container
        self._size = size
        self._readonly = readonly
        self._kwargs = kwargs
        self._closed = False

        # Track all connections
        # Note: _in_use uses id() keys because AbstractFileSystem.__eq__
        # compares by _fs_token: two instances to the same pool/container
        # compare equal, which breaks set membership.
        self._all: list[DFSFileSystem] = []
        self._available_queue: deque[DFSFileSystem] = deque()
        self._in_use: dict[int, DFSFileSystem] = {}
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)

        # Fork tracking: see _fork.py. The pooled filesystems poison
        # themselves, but the pool's own condition variable can also be
        # inherited in a locked state, so it needs replacing in the child.
        self._pid = os.getpid()
        self._fork_generation = _fork.generation()
        self._forked = False
        _fork.register(self)

        for _ in range(size):
            fs = DFSFileSystem(
                pool=pool,
                container=container,
                readonly=readonly,
                object_class=object_class,
                chunk_size=chunk_size,
                retry_count=retry_count,
                retry_backoff=retry_backoff,
                **kwargs,
            )
            self._all.append(fs)
            self._available_queue.append(fs)

    def acquire(self, timeout: float | None = None) -> DFSFileSystem:
        """Acquire a connection from the pool.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait. None means wait forever.
            Raises TimeoutError if exceeded.

        Raises
        ------
        RuntimeError
            If the pool is closed.
        TimeoutError
            If timeout is exceeded.
        """
        self._check_fork()
        with self._available:
            if self._closed:
                raise RuntimeError("pool is closed")

            end = time.monotonic() + timeout if timeout is not None else None

            while not self._available_queue:
                if self._closed:
                    raise RuntimeError("pool is closed")
                if end is not None:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for connection")
                    self._available.wait(timeout=remaining)
                else:
                    self._available.wait()

            fs = self._available_queue.popleft()
            self._in_use[id(fs)] = fs
            return fs

    def release(self, fs: DFSFileSystem) -> None:
        """Release a connection back to the pool.

        If the connection is dead (is_alive returns False), it is discarded
        instead of returned to the pool.
        """
        with self._available:
            if fs not in self._all:
                return  # Not from this pool, ignore
            if id(fs) not in self._in_use:
                return  # Already released (double-release), ignore
            del self._in_use[id(fs)]
            if self._closed:
                # Pool is closed: close this connection now
                try:
                    fs.close()
                except Exception:
                    pass
                return
            # Check if connection is still alive before returning to pool
            if not fs.is_alive:
                try:
                    fs.close()
                except Exception:
                    pass
                return
            self._available_queue.append(fs)
            self._available.notify()

    @contextmanager
    def connection(self, timeout: float | None = None) -> Generator[DFSFileSystem, None, None]:
        """Context manager that acquires and releases a connection."""
        fs = self.acquire(timeout=timeout)
        try:
            yield fs
        finally:
            self.release(fs)

    def health_check(self) -> dict[str, bool]:
        """Per-connection health, keyed ``connection_0``, ``connection_1``, ..."""
        # Snapshot connections under lock, then check each outside lock
        # so a slow health_check doesn't block acquire/release.
        with self._lock:
            connections = list(enumerate(self._all))
        results = {}
        for i, fs in connections:
            try:
                results[f"connection_{i}"] = bool(fs.health_check())
            except Exception:
                _logger.debug("health_check raised for connection_%d", i, exc_info=True)
                results[f"connection_{i}"] = False
        return results

    def close(self, force: bool = False) -> None:
        """Close the pool.

        Parameters
        ----------
        force : bool
            If False (default), only close idle connections. In-use
            connections are closed when released.
            If True, close all connections immediately (may interrupt
            active operations).
        """
        if self._forked or self._fork_generation != _fork.generation():
            # Inherited across a fork: every pooled connection is already
            # poisoned and closing them would hang. Drop the references.
            self._closed = True
            self._available_queue.clear()
            self._in_use.clear()
            self._all.clear()
            return

        with self._available:
            self._closed = True
            # Wake all threads blocked in acquire()
            self._available.notify_all()
            # Close available connections
            while self._available_queue:
                fs = self._available_queue.popleft()
                try:
                    fs.close()
                except Exception:
                    pass
            if force:
                # Close in-use connections immediately
                for fs in list(self._in_use.values()):
                    try:
                        fs.close()
                    except Exception:
                        pass
                self._in_use.clear()

    # ── Fork safety ──────────────────────────────────────────────────────

    def _poison_after_fork(self) -> None:
        """Mark the pool unusable in a forked child. Called by the fork hook.

        The pooled ``DFSFileSystem`` instances poison themselves via their own
        registration; this only has to deal with the pool's own condition
        variable, which may have been inherited held by a thread that no
        longer exists.
        """
        self._forked = True
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)

    def _check_fork(self) -> None:
        """Raise if the pool is being used from a forked child."""
        if self._forked or self._fork_generation != _fork.generation():
            raise _fork.fork_error(self._pid)

    def __enter__(self) -> DFSFileSystemPool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(force=True)

    def __del__(self) -> None:
        """Safety net: tear the pool down if close() was never called."""
        try:
            # During interpreter shutdown the _bindings module may already be
            # collected, and DFSFileSystem teardown calls into it. Imported
            # function-locally so this sees the live module, not a stale global.
            from daos_fsspec import _bindings as B

            if B is None or not hasattr(B, "dfs_release"):
                return
            self._close_from_finalizer()
        except Exception:
            pass

    def _close_from_finalizer(self) -> None:
        """Best-effort pool teardown that never blocks.

        Nobody can be waiting in ``acquire()`` while this runs: a waiting
        thread holds a reference to the pool, so the pool would not be being
        collected. That is why this does not need to notify the condition.
        """
        if self._forked or self._fork_generation != _fork.generation():
            # Inherited across a fork: every pooled connection is already
            # poisoned and closing them would hang. Drop the references.
            self._closed = True
            self._available_queue.clear()
            self._in_use.clear()
            self._all.clear()
            return

        if not self._lock.acquire(blocking=False):
            # Someone is mid-acquire/release. Leave the pool alone rather than
            # wait; the handles leak until process exit, as they did on this
            # path before.
            return
        try:
            self._closed = True
            connections = list(self._all)
            self._available_queue.clear()
            self._in_use.clear()
            self._all.clear()
        finally:
            self._lock.release()

        # Outside the pool lock: each connection takes its own lock, and each
        # declines to block in turn.
        for fs in connections:
            try:
                fs._close_from_finalizer()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"DFSFileSystemPool(pool={self._pool_label!r}, "
            f"container={self._cont_label!r}, "
            f"size={self._size})"
        )

    def status(self) -> dict[str, Any]:
        """Return pool status for debugging."""
        with self._lock:
            return {
                "pool": self._pool_label,
                "container": self._cont_label,
                "size": self._size,
                "available": len(self._available_queue),
                "in_use": len(self._in_use),
                "readonly": self._readonly,
                "closed": self._closed,
            }
