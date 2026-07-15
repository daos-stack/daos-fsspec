# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSFile — buffered file handle for DAOS DFS objects."""

from __future__ import annotations

import os
import stat as stat_mod
from typing import TYPE_CHECKING, Any

from fsspec.spec import AbstractBufferedFile

from daos_fsspec import _bindings as B

if TYPE_CHECKING:
    from daos_fsspec.core import DFSFileSystem


class DFSFile(AbstractBufferedFile):
    """Buffered file backed by a DFS object.

    Uses AbstractBufferedFile for read buffering and write buffering.
    All libdfs calls acquire the parent filesystem's lock.

    Note: O_APPEND is not supported by DAOS DFS. Use seek + write instead.
    """

    def __init__(
        self,
        fs: DFSFileSystem,
        path: str,
        mode: str = "rb",
        **kwargs: Any,
    ):
        from daos_fsspec.core import DFSFileSystem

        path = DFSFileSystem._strip_protocol(path)

        # Initialize cleanup-guard attributes first so __del__ is safe even if
        # construction aborts below (e.g. on the read-only guard).
        self._dfs_obj = None
        self._parent_obj = None
        self._dfs_closed = (
            False  # guards DFS handle cleanup (separate from AbstractBufferedFile._closed)
        )

        # Fail fast with a clean error instead of a raw DAOS error deep in
        # libdfs when the mount is closed or the open requests write access on
        # a read-only filesystem.
        fs._check_closed()
        if ("w" in mode or "a" in mode) and getattr(fs, "_readonly", False):
            raise PermissionError("filesystem is read-only")

        # Resolve parent and open/create the file
        parent, name = fs._resolve_path(path)
        self._parent_obj = parent

        try:
            if "r" in mode and "w" not in mode and "a" not in mode:
                # Read mode — file must exist
                with fs._lock:
                    self._dfs_obj, st = B.dfs_lookup_rel_path(fs._dfs, parent, name, os.O_RDONLY)
                size = st.size
            elif "w" in mode:
                # Write mode — create or truncate
                with fs._lock:
                    self._dfs_obj = B.dfs_open_file(
                        fs._dfs,
                        parent,
                        name,
                        stat_mod.S_IFREG | 0o644,
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                        oclass_id=getattr(fs, "_object_class", 0),
                        chunk_size=getattr(fs, "_chunk_size", 0),
                    )
                size = 0
            elif "a" in mode:
                # Append mode — create if needed, seek to end
                with fs._lock:
                    try:
                        self._dfs_obj, st = B.dfs_lookup_rel_path(fs._dfs, parent, name, os.O_RDWR)
                        size = st.size
                    except FileNotFoundError:
                        self._dfs_obj = B.dfs_open_file(
                            fs._dfs,
                            parent,
                            name,
                            stat_mod.S_IFREG | 0o644,
                            os.O_WRONLY | os.O_CREAT,
                            oclass_id=getattr(fs, "_object_class", 0),
                            chunk_size=getattr(fs, "_chunk_size", 0),
                        )
                        size = 0
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            super().__init__(fs, path, mode, size=size, **kwargs)
        except Exception:
            # Release handles on construction failure
            if self._dfs_obj is not None:
                try:
                    B.dfs_close(self._dfs_obj)
                except Exception:
                    pass
                self._dfs_obj = None
            fs._release_parent(parent)
            self._parent_obj = None
            raise

    def _fetch_range(self, start: int, end: int) -> bytes:
        """Read bytes [start, end) from the DFS object."""
        length = end - start
        if length <= 0:
            return b""
        if self._dfs_obj is None:
            raise RuntimeError("DFS object handle is closed")
        with self.fs._lock:
            return B.dfs_read(self.fs._dfs, self._dfs_obj, start, length)

    def _upload_chunk(self, final: bool = False) -> bool:
        """Write buffered data to DFS."""
        if self.buffer is None:
            return True

        data = self.buffer.getvalue()
        if not data:
            return True

        if self._dfs_obj is None:
            raise RuntimeError("DFS object handle is closed")
        with self.fs._lock:
            B.dfs_write(self.fs._dfs, self._dfs_obj, self.offset, data)

        return True

    def close(self) -> None:
        """Flush and release the DFS object handle.

        The ``finally`` block guarantees that DFS handles are released even
        if ``super().close()`` raises (e.g. from a failed flush).  This
        prevents handle leaks on I/O errors.
        """
        if self._dfs_closed:
            return
        self._dfs_closed = True
        # Let AbstractBufferedFile flush its buffers first.
        # AbstractBufferedFile.close() checks self._closed (not _dfs_closed)
        # and calls flush(force=True) before setting _closed = True.
        try:
            super().close()
        finally:
            # Release DFS object handle — always, even if flush failed.
            # Serialize the libdfs release against other ops on the mount.
            if self._dfs_obj is not None:
                with self.fs._lock:
                    B.dfs_close(self._dfs_obj)
                self._dfs_obj = None
            if self._parent_obj is not None:
                self.fs._release_parent(self._parent_obj)
                self._parent_obj = None

    def __del__(self) -> None:
        """Safety net: release handles if close() was never called."""
        if self._dfs_closed:
            return
        try:
            # During interpreter shutdown, the _bindings module may already
            # be garbage collected. Guard against AttributeError/segfault.
            if B is None or not hasattr(B, "dfs_close"):
                return
            # If the parent filesystem is already closed, the DFS handle is
            # invalid — don't try to release individual file handles.
            if getattr(self.fs, "_closed", True):
                self._dfs_closed = True
                return
            # Serialize the libdfs release against other ops, but never block
            # the GC/finalizer thread: if another thread holds the mount lock,
            # skip the release (the handle leaks until process exit, which is
            # safe) rather than risk hanging during interpreter shutdown.
            if not self.fs._lock.acquire(blocking=False):
                return
            try:
                if self._dfs_obj is not None:
                    B.dfs_close(self._dfs_obj)
                    self._dfs_obj = None
                if self._parent_obj is not None and self._parent_obj != self.fs._root:
                    B.dfs_release(self._parent_obj)
                    self._parent_obj = None
            finally:
                self.fs._lock.release()
        except Exception:
            pass
        finally:
            self._dfs_closed = True

    def __repr__(self) -> str:
        return f"DFSFile(path={self.path!r}, mode={self.mode!r})"
