# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSAsyncFileSystem — async fsspec filesystem for DAOS DFS containers.

Wraps DFSFileSystem with asyncio-compatible methods using asyncio.to_thread().
Requires Python 3.9+.

Architecture
------------
Uses an internal :class:`DFSFileSystemPool` to run multiple async operations
in parallel. Each DFSFileSystem instance holds an independent DFS handle with
its own lock, so operations on different handles don't contend. The pool size controls the maximum number of
concurrent operations.

This is a **compatibility wrapper**, not a true non-blocking implementation.
All DAOS operations run in a thread pool via ``asyncio.to_thread()``.
However, unlike a single-connection wrapper, multiple operations can execute
simultaneously on different handles.

For the DAOS C API, only ``dfs_read`` and ``dfs_write`` support non-blocking
mode (via ``daos_event_t``). All metadata operations (lookup, open, stat,
readdirplus, mkdir, remove, move) are always blocking. Therefore, a pool of
independent handles with per-handle locks is the correct design — it gives real parallelism
without the complexity of event queues.

Usage::

    import asyncio
    from daos_fsspec.async_core import DFSAsyncFileSystem

    async def main():
        fs = DFSAsyncFileSystem(pool="mypool", container="mycont", size=4)
        # Multiple reads run in parallel on different DFS handles
        results = await asyncio.gather(
            fs._cat_file("/data/shard-00000.parquet"),
            fs._cat_file("/data/shard-00001.parquet"),
            fs._cat_file("/data/shard-00002.parquet"),
        )
        await fs.close()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from typing import Any

from fsspec.asyn import AsyncFileSystem

from daos_fsspec.pool import DFSFileSystemPool


class DFSAsyncFileSystem(AsyncFileSystem):
    """Async fsspec filesystem backed by DAOS DFS.

    Uses an internal connection pool for parallel operations.
    Each connection holds an independent DFS handle with its own lock,
    so concurrent async operations on different connections don't contend.

    Parameters
    ----------
    pool : str
        DAOS pool label.
    container : str
        DAOS container label.
    size : int
        Number of DFS connections in the pool (default: 4).
        Controls max concurrency for async operations.
    readonly : bool
        Mount read-only (default: False).
    retry_count : int
        Number of retries for transient errors (default: 3).
    retry_backoff : float
        Initial backoff in seconds (default: 0.1).
    """

    protocol = "dfs"
    cachable = False

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
        super().__init__(**kwargs)
        self._pool = DFSFileSystemPool(
            pool=pool,
            container=container,
            size=size,
            readonly=readonly,
            object_class=object_class,
            chunk_size=chunk_size,
            retry_count=retry_count,
            retry_backoff=retry_backoff,
        )

    def _acquire(self):
        """Acquire a connection from the pool (context manager)."""
        return self._pool.connection()

    async def _ls(self, path: str, detail: bool = False, **kwargs: Any) -> list:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.ls, path, detail=detail, **kwargs)

    async def _info(self, path: str, **kwargs: Any) -> dict:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.info, path, **kwargs)

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.exists, path, **kwargs)

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.cat_file, path, start=start, end=end, **kwargs)

    async def _pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.pipe_file, path, value, **kwargs)

    async def _mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.mkdir, path, create_parents=create_parents, **kwargs)

    async def _makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.makedirs, path, exist_ok=exist_ok, **kwargs)

    async def _rm(
        self, path: str, recursive: bool = False, maxdepth: int | None = None, **kwargs: Any
    ) -> None:
        # Note: do NOT forward **kwargs — the sync rm() has a fixed signature
        # (path, recursive, maxdepth) and fsspec's _rm_file passes extras like
        # batch_size= which would raise TypeError.
        with self._acquire() as fs:
            await asyncio.to_thread(fs.rm, path, recursive=recursive, maxdepth=maxdepth)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.rm, path)

    async def _mv(self, path1: str, path2: str, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.mv, path1, path2, **kwargs)

    async def _get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        """Download a remote DFS file to a local path (streamed in chunks)."""
        with self._acquire() as fs:
            await asyncio.to_thread(fs.get_file, rpath, lpath, **kwargs)

    async def _put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        """Upload a local file to a remote DFS path (streamed in chunks)."""
        with self._acquire() as fs:
            await asyncio.to_thread(fs.put_file, lpath, rpath, **kwargs)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy one DFS file to another path within the same container.

        DAOS exposes no server-side single-file copy, so this reads the source
        and writes the destination in chunks. The transfer is done on one
        pooled connection to avoid coordinating offsets across two handles.
        """
        block_size = int(kwargs.pop("block_size", 16 * 1024 * 1024))

        def _copy_streaming() -> None:
            with self._acquire() as fs:
                with fs.open(path1, "rb") as src:
                    with fs.open(path2, "wb") as dst:
                        while True:
                            chunk = src.read(block_size)
                            if not chunk:
                                break
                            dst.write(chunk)

        await asyncio.to_thread(_copy_streaming)

    async def _symlink(self, path: str, target: str, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.symlink, path, target, **kwargs)

    async def _readlink(self, path: str) -> str:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.readlink, path)

    async def _touch(self, path: str, truncate: bool = True, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.touch, path, truncate=truncate, **kwargs)

    async def _walk(
        self, path: str, maxdepth: int | None = None, topdown: bool = True, **kwargs: Any
    ):
        """Async walk — yields (path, dirs, files) tuples."""
        with self._acquire() as fs:
            for item in await asyncio.to_thread(
                lambda: list(fs.walk(path, maxdepth=maxdepth, topdown=topdown, **kwargs))
            ):
                yield item

    async def _cat_ranges(
        self,
        paths: list[str],
        starts: list[int] | int,
        ends: list[int] | int,
        max_gap: int | None = None,
        on_error: str = "return",
        **kwargs: Any,
    ) -> list[bytes | Exception]:
        if not isinstance(paths, list):
            raise TypeError("paths must be a list")
        if on_error not in {"return", "raise"}:
            raise ValueError("on_error must be 'return' or 'raise'")

        n = len(paths)
        if isinstance(starts, (int, type(None))):
            starts = [starts] * n
        if isinstance(ends, (int, type(None))):
            ends = [ends] * n
        if len(starts) != n or len(ends) != n:
            raise ValueError("starts and ends must be scalars or lists matching paths length")

        async def _read_one(p: str, s: int | None, e: int | None) -> bytes | Exception:
            try:
                with self._acquire() as fs:
                    return await asyncio.to_thread(fs.cat_file, p, start=s, end=e, **kwargs)
            except Exception as exc:
                if on_error == "return":
                    return exc
                raise

        # Run the range reads concurrently. Each read borrows its own pooled
        # connection (with its own lock), so the to_thread() calls parallelize
        # across handles; gather preserves input order in the returned list.
        return list(
            await asyncio.gather(*[_read_one(p, s, e) for p, s, e in zip(paths, starts, ends)])
        )

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        **kwargs: Any,
    ) -> list[str] | dict[str, Any]:
        with self._acquire() as fs:
            return await asyncio.to_thread(
                fs.find, path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs
            )

    async def _head(self, path: str, size: int = 1024, **kwargs: Any) -> bytes:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.head, path, size=size, **kwargs)

    async def _tail(self, path: str, size: int = 1024, **kwargs: Any) -> bytes:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.tail, path, size=size, **kwargs)

    async def _getxattr(self, path: str, attr_name: str, **kwargs: Any) -> bytes:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.getxattr, path, attr_name, **kwargs)

    async def _setxattr(
        self, path: str, attr_name: str, value: bytes, flags: int = 0, **kwargs: Any
    ) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.setxattr, path, attr_name, value, flags=flags, **kwargs)

    async def _listxattr(self, path: str, **kwargs: Any) -> list[str]:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.listxattr, path, **kwargs)

    async def _removexattr(self, path: str, attr_name: str, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.removexattr, path, attr_name, **kwargs)

    async def _du(self, path: str, total: bool = False, maxdepth: int | None = None, **kwargs: Any):
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.du, path, total=total, maxdepth=maxdepth, **kwargs)

    async def _checksum(self, path: str, **kwargs: Any) -> str:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.checksum, path, **kwargs)

    async def _created(self, path: str, **kwargs: Any) -> float:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.created, path, **kwargs)

    async def _modified(self, path: str, **kwargs: Any) -> float:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.modified, path, **kwargs)

    async def pool_space(self):
        """Query the DAOS pool storage space."""
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.pool_space)

    async def _chown(
        self, path: str, uid: int, gid: int, follow_links: bool = True, **kwargs: Any
    ) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.chown, path, uid, gid, follow_links=follow_links, **kwargs)

    async def _access(self, path: str, mode: int = 0, **kwargs: Any) -> bool:
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.access, path, mode=mode, **kwargs)

    async def mount_attr(self):
        """Query the DFS mount attributes."""
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.mount_attr)

    async def _obj_info(self, path: str, **kwargs: Any):
        """Query object info (oclass, chunk_size, OID) for a file."""
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.obj_info, path, **kwargs)

    async def _truncate(self, path: str, size: int = 0, **kwargs: Any) -> None:
        with self._acquire() as fs:
            await asyncio.to_thread(fs.truncate, path, size=size, **kwargs)

    async def _find_parallel(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        num_workers: int = 4,
        **kwargs: Any,
    ):
        """Async parallel find using anchor split."""
        with self._acquire() as fs:
            return await asyncio.to_thread(
                fs.find_parallel,
                path,
                maxdepth=maxdepth,
                withdirs=withdirs,
                detail=detail,
                num_workers=num_workers,
                **kwargs,
            )

    async def _suggest_oclass(self, hint: str, **kwargs: Any) -> int:
        """Suggest a DAOS object class from a hint string."""
        with self._acquire() as fs:
            return await asyncio.to_thread(fs.suggest_oclass, hint, **kwargs)

    async def _set_prefix(self, prefix: str, **kwargs: Any) -> None:
        """Set a prefix to trim from all DFS paths."""
        with self._acquire() as fs:
            await asyncio.to_thread(fs.set_prefix, prefix, **kwargs)

    async def close(self) -> None:
        """Close all connections in the pool."""
        await asyncio.to_thread(self._pool.close, force=True)

    async def health_check(self) -> dict[str, bool]:
        """Check health of all connections in the pool."""
        return await asyncio.to_thread(self._pool.health_check)

    def __repr__(self) -> str:
        return f"DFSAsyncFileSystem({self._pool!r})"
