# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSFileSystemPool — connection pool for DAOS DFS.

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

import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from daos_fsspec.core import DFSFileSystem


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
        # compares by _fs_token — two instances to the same pool/container
        # compare equal, which breaks set membership.
        self._all: list[DFSFileSystem] = []
        self._available_queue: deque[DFSFileSystem] = deque()
        self._in_use: dict[int, DFSFileSystem] = {}
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)

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
                # Pool is closed — close this connection now
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
        """Check health of all connections in the pool."""
        # Snapshot connections under lock, then check each outside lock
        # so a slow health_check doesn't block acquire/release.
        with self._lock:
            connections = list(enumerate(self._all))
        results = {}
        for i, fs in connections:
            results[f"connection_{i}"] = fs.health_check()
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

    def __enter__(self) -> DFSFileSystemPool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(force=True)

    def __del__(self) -> None:
        try:
            # During interpreter shutdown, _bindings module may be garbage
            # collected. DFSFileSystem.close() calls C functions via _bindings,
            # which would segfault. Skip cleanup if _bindings is gone.
            from daos_fsspec import _bindings as B

            if B is None or not hasattr(B, "dfs_release"):
                return
            self.close(force=True)
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
