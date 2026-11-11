# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSFile: buffered file handle for DAOS DFS objects."""

from __future__ import annotations

import logging
import os
import stat as stat_mod
import sys
import weakref
from typing import TYPE_CHECKING, Any

from fsspec.spec import AbstractBufferedFile

from daos_fsspec import _bindings as B

# same_handle is imported directly, not reached through `B`. It is pure
# attribute comparison with no libdaos dependency, so it never needs mocking --
# and routing it through the patched `B` would be actively harmful: a MagicMock
# returns a truthy Mock for any call, so `if B.same_handle(...): return` would
# skip every handle release in every mocked test. Three tests caught exactly
# that. Mocking exists here to avoid dlopen; this function does not dlopen.
from daos_fsspec._bindings import same_handle

if TYPE_CHECKING:
    from daos_fsspec.core import DFSFileSystem


_logger = logging.getLogger(__name__)


class DFSFile(AbstractBufferedFile):
    """Buffered file backed by a DFS object."""

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
        self.fs = fs
        self.mode = mode
        self.path = path
        self._dfs_obj = None
        self._parent_obj = None
        # The mount generation these handles belong to. It is set atomically
        # with parent resolution below; sampling it here would let a reconnect
        # land between the sample and acquisition of the actual handles.
        # A file can easily outlive a reconnect: nothing in the read/write path
        # retries, but another thread doing ls() can hit a retryable error and
        # reconnect while this file is open. That umounts the filesystem our
        # handles came from, so closing afterwards would hand libdfs freed
        # memory. Compared before I/O and teardown; mismatch means fail the I/O
        # and drop references rather than touching the old mount.
        self._mount_generation = -1
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
        parent, name, generation = fs._resolve_path(path)
        self._mount_generation = generation
        self._parent_obj = parent

        with fs._lock.read():
            fs._assert_same_mount(generation, "opening a file")
            fs._open_files[id(self)] = weakref.ref(self)

        try:
            if "r" in mode and "w" not in mode and "a" not in mode:
                # Read mode: file must exist
                with fs._lock.read():
                    fs._assert_same_mount(generation, "opening a file")
                    self._dfs_obj, st = B.dfs_lookup_rel_path(fs._dfs, parent, name, os.O_RDONLY)
                size = st.size
            elif "w" in mode:
                # Write mode: create or truncate
                with fs._lock.read():
                    fs._assert_same_mount(generation, "opening a file")
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
                # Append mode: create if needed, seek to end
                with fs._lock.read():
                    fs._assert_same_mount(generation, "opening a file")
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
            # Where the first buffered write lands. For "ab" that is the end of
            # the existing file; see _initiate_upload for why recording it is
            # not enough on its own.
            self._write_offset = size if "a" in mode else 0
            super().__init__(fs, path, mode, size=size, **kwargs)
        except BaseException:
            # Preserve the opening error. Failed cleanup remains registered;
            # mount teardown must not retry a possibly destructive release.
            self._dfs_closed = True
            try:
                if self._dfs_obj is not None:
                    fs._release_object(self._dfs_obj, generation, close=True)
                    self._dfs_obj = None
                fs._release_parent(parent, generation)
                self._parent_obj = None
                fs._open_files.pop(id(self), None)
            except Exception:
                _logger.warning("cleanup after failed file construction failed", exc_info=True)
            raise

    def _handles_are_current(self) -> bool:
        """True if our handles still belong to the filesystem's live mount."""
        return self._mount_generation == getattr(self.fs, "_mount_generation", None)

    def _initiate_upload(self) -> None:
        """Seed the write offset before the first chunk is written."""
        if "a" in self.mode:
            self.offset = self._write_offset

    def _fetch_range(self, start: int, end: int) -> bytes:
        """Read bytes [start, end) from the DFS object.

        Delegates to the filesystem's pipelined reader, which overlaps the
        chunk reads for large ranges and falls back to a single call below the
        threshold. No lock is held here: _read_pipelined takes the shared lock
        per libdfs call, and holding it around a wait on executor workers is
        the pattern that risks deadlocking against a waiting writer.
        """
        length = end - start
        if length <= 0:
            return b""
        if self._dfs_obj is None:
            raise RuntimeError("DFS object handle is closed")
        self.fs._check_fork()
        # Annotated because AbstractBufferedFile.fs is untyped, so the call
        # would otherwise return Any out of a bytes-declared method.
        data: bytes = self.fs._read_pipelined(
            self._dfs_obj, start, length, generation=self._mount_generation
        )
        return data

    def _upload_chunk(self, final: bool = False) -> bool:
        """Write buffered data to DFS."""
        if self.buffer is None:
            return True

        data = self.buffer.getvalue()
        if not data:
            return True

        if self._dfs_obj is None:
            raise RuntimeError("DFS object handle is closed")
        self.fs._check_fork()
        with self.fs._lock.read():
            self.fs._assert_same_mount(self._mount_generation, "writing an open file")
            B.dfs_write(self.fs._dfs, self._dfs_obj, self.offset, data)

        # A buffered write changes this file's size, which ls() caches. Without
        # this, writing through fs.open(..., "wb") would leave a stale size (or
        # omit a newly created file entirely) for up to the cache TTL, while
        # pipe_file; which invalidates; would look correct. Invalidating per
        # chunk rather than only in close() keeps the cache honest even for a
        # file that is still being written.
        invalidate = getattr(self.fs, "_invalidate_listing", None)
        if invalidate is not None:
            invalidate(self.path)

        return True

    def close(self) -> None:
        """Flush and release the DFS object handle.

        The ``finally`` block guarantees that DFS handles are released even
        if ``super().close()`` raises (e.g. from a failed flush).  This
        prevents handle leaks on I/O errors.
        """
        if getattr(self, "_write_invalidated", False):
            raise OSError("buffered writes were discarded by filesystem teardown")
        if self._dfs_closed:
            return
        if getattr(self.fs, "_closed", True) and self._handles_are_current():
            # The parent's teardown owns these handles now. Keep them visible
            # until it releases them; dropping them here would hide a leak.
            self._closed = True
            return
        self._dfs_closed = True
        # Inherited across a fork: the handles are unusable and flushing or
        # releasing would hang in libdfs. Drop them and let them leak until
        # this child exits: the parent still owns the real handles.
        if getattr(self.fs, "_forked", False):
            self._dfs_obj = None
            self._parent_obj = None
            return
        # The filesystem reconnected while this file was open: the mount these
        # handles came from is already gone, so flushing or closing them would
        # be a use-after-free. Drop the references. Buffered writes are lost,
        # which is unavoidable; the mount they were destined for no longer
        # exists; and the reconnect already released the handles.
        if getattr(self.fs, "_closed", True) or not self._handles_are_current():
            self._dfs_obj = None
            self._parent_obj = None
            return
        # Let AbstractBufferedFile flush its buffers first.
        # AbstractBufferedFile.close() checks self._closed (not _dfs_closed)
        # and calls flush(force=True) before setting _closed = True.
        try:
            super().close()
        finally:
            primary_error = sys.exc_info()[0]
            try:
                # Release DFS object handle: always, even if flush failed.
                # Serialize the libdfs release against other ops on the mount.
                if self._dfs_obj is not None:
                    self.fs._release_object(self._dfs_obj, self._mount_generation, close=True)
                    self._dfs_obj = None
                if self._parent_obj is not None:
                    self.fs._release_parent(self._parent_obj, self._mount_generation)
                    self._parent_obj = None
                self.fs._open_files.pop(id(self), None)
            except Exception:
                if primary_error is None:
                    raise
                _logger.warning("file cleanup failed after flush failure", exc_info=True)

    def _discard_for_teardown(self) -> None:
        """Release an open file under its parent's exclusive teardown lock."""
        if self._dfs_closed and (self._dfs_obj is not None or self._parent_obj is not None):
            raise RuntimeError("file cleanup previously failed; refusing retry")
        self._write_invalidated = (
            ("w" in self.mode or "a" in self.mode)
            and getattr(self, "buffer", None) is not None
            and self.buffer.tell() > 0
        )
        self._dfs_closed = self._closed = True
        if "w" in self.mode or "a" in self.mode:
            _logger.warning("Discarding unclosed writable file %s without flushing", self.path)
        if self._handles_are_current():
            if self._dfs_obj is not None:
                obj, self._dfs_obj = self._dfs_obj, None
                B.dfs_close(obj)
            if self._parent_obj is not None:
                parent, self._parent_obj = self._parent_obj, None
                if not same_handle(parent, self.fs._root):
                    B.dfs_release(parent)
        self.fs._open_files.pop(id(self), None)

    def __del__(self) -> None:
        """Safety net: release handles if close() was never called."""
        if self._dfs_closed:
            return
        try:
            # During interpreter shutdown, the _bindings module may already
            # be garbage collected. Guard against AttributeError/segfault.
            if B is None or not hasattr(B, "dfs_close"):
                return
            # Only release through the library that opened the handle; see
            # DFSFileSystem._close_from_finalizer for why this is not academic.
            if getattr(self.fs, "_bindings", None) is not B:
                return
            # If the parent filesystem is already closed, the DFS handle is
            # invalid: don't try to release individual file handles.
            if getattr(self.fs, "_closed", True):
                self._dfs_closed = True
                return
            # Same for a fork: calling into libdfs here would hang.
            if getattr(self.fs, "_forked", False):
                self._dfs_closed = True
                return
            # The mount our handles came from has been replaced by a reconnect;
            # releasing into the new one would be a use-after-free.
            if not self._handles_are_current():
                self._dfs_closed = True
                self._dfs_obj = None
                self._parent_obj = None
                return
            # Releasing a handle only needs the shared lock: it must not race
            # teardown, but it does not conflict with other I/O. Never block
            # the GC/finalizer thread: if teardown holds the lock exclusively,
            # skip the release (the handle leaks until process exit, which is
            # safe) rather than risk hanging during interpreter shutdown.
            if not self.fs._lock.acquire_read(blocking=False):
                return
            try:
                if (
                    getattr(self.fs, "_closed", True)
                    or getattr(self.fs, "_forked", False)
                    or not self._handles_are_current()
                ):
                    self._dfs_obj = None
                    self._parent_obj = None
                    return
                if self._dfs_obj is not None:
                    B.dfs_close(self._dfs_obj)
                    self._dfs_obj = None
                if self._parent_obj is not None and not same_handle(
                    self._parent_obj, self.fs._root
                ):
                    B.dfs_release(self._parent_obj)
                    self._parent_obj = None
                self.fs._open_files.pop(id(self), None)
            finally:
                self.fs._lock.release_read()
        except Exception:
            pass
        finally:
            self._dfs_closed = True

    def __repr__(self) -> str:
        return f"DFSFile(path={self.path!r}, mode={self.mode!r})"
