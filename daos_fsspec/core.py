# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSFileSystem — fsspec filesystem for DAOS DFS containers.

Usage::

    import fsspec
    fs = fsspec.filesystem("dfs", pool="mypool", container="mycont")
    files = fs.ls("/")
    data = fs.cat("/path/to/file")

    # URI shorthand (after entry point registration):
    df = pd.read_parquet("dfs://mypool/mycont/data.parquet")

    # Context manager:
    with DFSFileSystem(pool="p", container="c") as fs:
        data = fs.cat("/file")
"""

from __future__ import annotations

import ctypes
import logging
import os
import stat as stat_mod
import threading
import time
from typing import Any, Callable, TypeVar

from fsspec.spec import AbstractFileSystem

from daos_fsspec import _bindings as B
from daos_fsspec.dfs_file import DFSFile

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# DFS dfs_lookup with O_RDONLY/O_RDWR returns EINVAL on symlinks.
# O_NOFOLLOW returns the symlink object itself without following.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0x20000)

# Default retry configuration
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_BACKOFF = 0.1  # seconds

# DFS limits (from daos_fs.h)
DFS_MAX_PATH = 4096
DFS_MAX_NAME = 255
_DEFAULT_MAX_MKDIR_DEPTH = 100


class DFSFileSystem(AbstractFileSystem):
    """fsspec filesystem backed by DAOS DFS (POSIX-like containers).

    Parameters
    ----------
    pool : str
        DAOS pool label (e.g., "mypool").
    container : str
        DAOS container label (e.g., "dataset-sql-v2").
    readonly : bool
        Mount the DFS namespace read-only (default: False).
    object_class : int
        Default DAOS object class ID for new files (default: 0 = container default).
        Common values: 0 (default), or specific OC_S* / OC_EC* IDs.
    chunk_size : int
        Default chunk/stripe size in bytes for new files (default: 0 = 1MB default).
        Common values: 0 (default 1MB), 4194304 (4MB), 16777216 (16MB).
    retry_count : int
        Number of retries for transient errors (default: 3).
    retry_backoff : float
        Initial backoff in seconds between retries (default: 0.1).
        Doubles on each retry (exponential backoff).
    """

    protocol = "dfs"
    cachable = False  # We manage our own connection lifecycle

    def __init__(
        self,
        pool: str,
        container: str,
        readonly: bool = False,
        object_class: int = 0,
        chunk_size: int = 0,
        retry_count: int = _DEFAULT_RETRY_COUNT,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._pool_label = pool
        self._cont_label = container
        self._readonly = readonly
        self._object_class = object_class
        self._chunk_size = chunk_size
        # Reentrant so helpers that release handles under the lock (e.g.
        # _release_parent, _resolve_path cleanup) can be safely called from
        # both locked and unlocked contexts without risk of self-deadlock.
        self._lock = threading.RLock()
        self._closed = False

        # DAOS handles (populated by _connect, cleared by close/_reconnect)
        self._dfs: Any = None
        self._poh: Any = None
        self._coh: Any = None
        self._root: Any = None

        # Resilience configuration
        self._retry_count = retry_count
        self._retry_backoff = retry_backoff
        self._last_activity: float = 0.0

        # Lazy-initialized executor for cat_ranges (thread-safe init)
        self._cat_ranges_executor = None
        self._executor_lock = threading.Lock()

        # Validate labels
        if not pool or not isinstance(pool, str):
            raise ValueError("pool must be a non-empty string")
        if not container or not isinstance(container, str):
            raise ValueError(
                "container must be a non-empty string. "
                "If using a URI, ensure it has the form dfs://pool/container/path"
            )
        if len(pool) > 127:  # DAOS_PROP_LABEL_MAX_LEN
            raise ValueError(f"pool label too long (max 127 chars): {len(pool)}")
        if len(container) > 127:
            raise ValueError(f"container label too long (max 127 chars): {len(container)}")

        # Connect to DAOS
        self._uses_dfs_connect = False
        self._connect()

    def _connect(self) -> None:
        """Establish DAOS connection.

        Uses dfs_connect() when available (single-call API, DAOS 2.6+),
        otherwise falls back to the 4-step: daos_init → pool_connect → cont_open → dfs_mount.
        """
        if B._HAS_DFS_CONNECT:
            self._connect_via_dfs_connect()
        else:
            self._connect_via_lowlevel()

    def _connect_via_dfs_connect(self) -> None:
        """Connect using the high-level dfs_connect() API (one call: pool+cont+mount)."""
        _, libdfs = B._ensure_libs()
        B.daos_init()
        # dfs_connect() requires dfs_init() to have been called
        libdfs.dfs_init()
        flags = os.O_RDONLY if self._readonly else os.O_RDWR
        dfs = B.dfs_p()
        rc = libdfs.dfs_connect(
            self._pool_label.encode(),
            None,  # sys (default)
            self._cont_label.encode(),
            flags,
            None,  # dfs_attr_t* (use defaults)
            ctypes.byref(dfs),  # type: ignore[arg-type]
        )
        B._check(rc, f"dfs_connect({self._pool_label}, {self._cont_label})")
        self._dfs = dfs
        self._poh = None  # not available from dfs_connect; use dfs_pool_get when needed
        self._coh = None
        self._uses_dfs_connect = True
        try:
            self._root, _ = B.dfs_lookup_path(
                self._dfs, "/", os.O_RDONLY if self._readonly else os.O_RDWR
            )
        except Exception:
            try:
                libdfs.dfs_disconnect(self._dfs)
            except Exception:
                pass
            self._dfs = None
            raise
        self._last_activity = time.monotonic()
        _logger.debug(
            "Connected via dfs_connect pool=%s container=%s", self._pool_label, self._cont_label
        )

    def _connect_via_lowlevel(self) -> None:
        """Connect using the low-level 4-step API (legacy path)."""
        B.daos_init()
        self._poh = B.pool_connect(
            self._pool_label, flags=B.DAOS_PC_RO if self._readonly else B.DAOS_PC_RW
        )
        try:
            self._coh = B.cont_open(
                self._poh,
                self._cont_label,
                flags=B.DAOS_COO_RO if self._readonly else B.DAOS_COO_RW,
            )
        except Exception:
            try:
                B.pool_disconnect(self._poh)
            except Exception:
                pass
            self._poh = None
            raise
        try:
            self._dfs = B.dfs_mount_fs(self._poh, self._coh, readonly=self._readonly)
        except Exception:
            try:
                B.cont_close(self._coh)
            except Exception:
                pass
            try:
                B.pool_disconnect(self._poh)
            except Exception:
                pass
            self._coh = None
            self._poh = None
            raise
        try:
            # Root object handle (kept open for the lifetime of the mount)
            self._root, _ = B.dfs_lookup_path(
                self._dfs, "/", os.O_RDONLY if self._readonly else os.O_RDWR
            )
        except Exception:
            try:
                B.dfs_umount(self._dfs)
            except Exception:
                pass
            try:
                B.cont_close(self._coh)
            except Exception:
                pass
            try:
                B.pool_disconnect(self._poh)
            except Exception:
                pass
            self._dfs = None
            self._coh = None
            self._poh = None
            raise
        self._uses_dfs_connect = False
        self._last_activity = time.monotonic()
        _logger.debug(
            "Connected via lowlevel pool=%s container=%s", self._pool_label, self._cont_label
        )

    def _reconnect(self) -> None:
        """Tear down and re-establish the DAOS connection.

        Sets ``self._closed = True`` during teardown so other threads see
        invalid handles immediately.  If ``_connect()`` fails, the
        filesystem remains closed — the caller will see a ``RuntimeError``
        on the next operation.
        """
        _logger.warning(
            "Reconnecting to DAOS pool=%s container=%s", self._pool_label, self._cont_label
        )
        with self._lock:
            # Mark closed during teardown so other threads see it immediately
            self._closed = True

            # Close old connection (ignore errors)
            if getattr(self, "_root", None) is not None:
                try:
                    B.dfs_release(self._root)
                except Exception:
                    pass
            if getattr(self, "_uses_dfs_connect", False) and B._HAS_DFS_CONNECT:
                try:
                    _, libdfs = B._ensure_libs()
                    libdfs.dfs_disconnect(self._dfs)
                except Exception:
                    pass
            else:
                for attr in ["_dfs", "_coh", "_poh"]:
                    handle = getattr(self, attr, None)
                    if handle is not None:
                        try:
                            if attr == "_dfs":
                                B.dfs_umount(handle)
                            elif attr == "_coh":
                                B.cont_close(handle)
                            elif attr == "_poh":
                                B.pool_disconnect(handle)
                        except Exception:
                            pass

            self._root = None
            self._dfs = None
            self._coh = None
            self._poh = None

            # Re-establish — stays closed if this fails
            self._connect()
            self._closed = False

    def _is_retryable(self, exc: Exception) -> bool:
        """Return True if the exception is a transient DAOS error."""
        if isinstance(exc, B.DaosError):
            return exc.is_retryable
        if isinstance(exc, OSError):
            return exc.errno in B.RETRYABLE_ERRORS
        return False

    def _retry(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Execute fn with retry logic for transient errors.

        Only retries on transient DAOS errors (DER_UNREACH, DER_HG, etc.).
        Non-retryable errors are raised immediately.
        """
        for attempt in range(self._retry_count + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not self._is_retryable(exc) or attempt == self._retry_count:
                    raise
                backoff = self._retry_backoff * (2**attempt)
                _logger.warning(
                    "Retryable error on attempt %d/%d: %s. Retrying in %.2fs...",
                    attempt + 1,
                    self._retry_count + 1,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                # Try reconnecting on connection-level errors
                if isinstance(exc, B.DaosError) and exc.rc in (
                    B.DER_UNREACH,
                    B.DER_HG,
                    B.DER_NO_HDL,
                    B.DER_TIMEDOUT,
                ):
                    try:
                        self._reconnect()
                    except Exception as reconnect_exc:
                        _logger.error(
                            "Reconnect failed on attempt %d: %s",
                            attempt + 1,
                            reconnect_exc,
                        )
                        # Reconnect failed — raise the original error with
                        # context, not the opaque "DFSFileSystem is closed".
                        raise exc from reconnect_exc
        raise AssertionError("unreachable")

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> DFSFileSystem:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Health / status ─────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check if the DAOS connection is alive.

        Returns True if the connection is healthy, False otherwise.
        """
        if self._closed:
            return False
        try:
            with self._lock:
                B.dfs_stat(self._dfs, self._root)
            self._last_activity = time.monotonic()
            return True
        except Exception:
            return False

    @property
    def is_alive(self) -> bool:
        """Return True if the connection appears to be alive."""
        return not self._closed

    def __repr__(self) -> str:
        return (
            f"DFSFileSystem(pool={self._pool_label!r}, "
            f"container={self._cont_label!r}, "
            f"readonly={self._readonly})"
        )

    def __reduce__(self):
        raise TypeError(
            "DFSFileSystem cannot be pickled. DAOS handles do not survive "
            "process boundaries. Each process must create its own instance. "
            "For multiprocessing DataLoaders, use a worker_init_fn that "
            "creates a new DFSFileSystem per worker."
        )

    def status(self) -> dict[str, Any]:
        """Return connection status for debugging."""
        return {
            "pool": self._pool_label,
            "container": self._cont_label,
            "readonly": self._readonly,
            "object_class": self._object_class,
            "chunk_size": self._chunk_size,
            "closed": self._closed,
            "retry_count": self._retry_count,
            "retry_backoff": self._retry_backoff,
            "last_activity": self._last_activity,
        }

    # ── URI parsing ──────────────────────────────────────────────────────

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        """Parse dfs://pool/container/path → /path."""
        if path.startswith("dfs://"):
            path = path[len("dfs://") :]
            parts = path.split("/", 2)
            if len(parts) >= 3:
                return cls._validate_path("/" + parts[2])
            return "/"
        if not path.startswith("/"):
            path = "/" + path
        return cls._validate_path(path)

    @classmethod
    def _get_kwargs_from_urls(cls, path: str) -> dict[str, Any]:
        """Extract pool and container from dfs://pool/container/... URI."""
        if path.startswith("dfs://"):
            rest = path[len("dfs://") :]
            parts = rest.split("/", 2)
            result: dict[str, Any] = {}
            if len(parts) >= 1 and parts[0]:
                result["pool"] = parts[0]
            if len(parts) >= 2 and parts[1]:
                result["container"] = parts[1]
            return result
        return {}

    # ── Path resolution (handle-leak-safe) ───────────────────────────────

    def _check_closed(self) -> None:
        """Raise if the filesystem is closed."""
        if self._closed:
            raise RuntimeError("DFSFileSystem is closed")

    @staticmethod
    def _check_not_root(path: str, operation: str) -> None:
        """Raise if an operation that requires a directory entry targets root."""
        if path == "/":
            raise ValueError(f"cannot {operation} the DFS root")

    def _resolve_path(self, path: str) -> tuple[Any, str]:
        """Resolve a path into (parent_obj, basename).

        For "/", returns (root_obj, "").
        For "/a/b/c", walks the tree to return (obj_for_/a/b, "c").

        All intermediate handles are released; only the final parent
        is kept alive (caller must release if != root).
        """
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"
        if path == "/":
            return self._root, ""

        parts = [p for p in path.split("/") if p]
        if len(parts) == 1:
            return self._root, parts[0]

        # Walk to parent, releasing intermediates as we go.
        # On error, release all intermediates to prevent handle leaks.
        parent = self._root
        intermediates: list[Any] = []
        lookup_flags = os.O_RDONLY if self._readonly else os.O_RDWR
        try:
            for part in parts[:-1]:
                with self._lock:
                    child, _ = B.dfs_lookup_rel_path(self._dfs, parent, part, lookup_flags)
                intermediates.append(child)
                parent = child
        except Exception:
            with self._lock:
                for obj in intermediates:
                    if obj != self._root:
                        B.dfs_release(obj)
            raise

        # Release all intermediates except the final parent
        with self._lock:
            for obj in intermediates[:-1]:
                if obj != self._root:
                    B.dfs_release(obj)

        return parent, parts[-1]

    def _release_parent(self, parent: Any) -> None:
        """Release a parent handle if it's not the root.

        Acquires the lock so the libdfs release is serialized against other
        operations on the same mount (libdfs handles are not thread-safe).
        """
        if parent != self._root:
            with self._lock:
                B.dfs_release(parent)

    @staticmethod
    def _validate_path(path: str) -> str:
        """Validate and normalize a DFS path.

        Raises ValueError if the path is invalid.
        """
        if "\x00" in path:
            raise ValueError("path must not contain NUL bytes")
        if len(path) > DFS_MAX_PATH:
            raise ValueError(f"path too long ({len(path)} > {DFS_MAX_PATH}): {path[:80]}...")
        # Validate individual components
        parts = path.split("/")
        for part in parts:
            if part and len(part) > DFS_MAX_NAME:
                raise ValueError(
                    f"path component too long ({len(part)} > {DFS_MAX_NAME}): {part[:80]}..."
                )
        return path

    # ── Directory listing ────────────────────────────────────────────────

    def ls(self, path: str, detail: bool = False, **kwargs: Any) -> list:
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _ls_inner() -> list[B.DirEntry]:
            with self._lock:
                if path == "/":
                    obj = self._root
                    need_release = False
                else:
                    obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                    need_release = True

                try:
                    entries = B.dfs_readdirplus(self._dfs, obj)
                finally:
                    if need_release:
                        B.dfs_release(obj)
            return entries

        entries = self._retry(_ls_inner)
        self._last_activity = time.monotonic()

        result: list[Any] = []
        for entry in entries:
            full_path = path.rstrip("/") + "/" + entry.name
            if full_path.startswith("//"):
                full_path = full_path[1:]
            if detail:
                if entry.stat:
                    result.append(
                        {
                            "name": full_path,
                            "size": entry.stat.size,
                            "type": entry.stat.type_name,
                            "mode": entry.stat.mode,
                            "mtime": entry.stat.mtime,
                            "atime": entry.stat.atime,
                            "uid": entry.stat.uid,
                            "gid": entry.stat.gid,
                        }
                    )
                else:
                    result.append(
                        {
                            "name": full_path,
                            "size": 0,
                            "type": "file",
                        }
                    )
            else:
                result.append(full_path)

        return result

    # ── File info / stat ─────────────────────────────────────────────────

    def info(self, path: str, **kwargs: Any) -> dict:
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _info_inner() -> B.StatResult:
            with self._lock:
                if path == "/":
                    st = B.dfs_stat(self._dfs, self._root)
                else:
                    # O_NOFOLLOW to correctly report symlinks as symlinks
                    obj, st = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                    B.dfs_release(obj)
            return st

        st = self._retry(_info_inner)
        self._last_activity = time.monotonic()

        return {
            "name": path,
            "size": st.size,
            "type": st.type_name,
            "mode": st.mode,
            "mtime": st.mtime,
            "atime": st.atime,
            "uid": st.uid,
            "gid": st.gid,
        }

    def exists(self, path: str, **kwargs: Any) -> bool:
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _exists_inner() -> bool:
            with self._lock:
                if path == "/":
                    return True
                # O_NOFOLLOW handles both regular files/dirs AND symlinks.
                # O_RDONLY on a symlink returns EINVAL in DFS.
                obj, _ = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                B.dfs_release(obj)
            return True

        try:
            result = self._retry(_exists_inner)
            self._last_activity = time.monotonic()
            return result
        except FileNotFoundError:
            return False
        except B.DaosError as exc:
            if exc.rc in (B.DER_NONEXIST, B.DER_NO_ENTRY):
                return False
            raise

    def lexists(self, path: str, **kwargs: Any) -> bool:
        """Like exists(), but does not follow symlinks.

        In DFS, dfs_lookup with O_NOFOLLOW already returns the symlink
        itself, so this is equivalent to exists().
        """
        return self.exists(path, **kwargs)

    def modified(self, path: str) -> float:
        """Return the modification time of a path as a Unix timestamp."""
        info = self.info(path)
        return float(info["mtime"])

    def created(self, path: str) -> float:
        """Return the creation time (ctime on POSIX) as a Unix timestamp."""
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _created_inner() -> B.StatResult:
            with self._lock:
                if path == "/":
                    st = B.dfs_stat(self._dfs, self._root)
                else:
                    obj, st = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                    B.dfs_release(obj)
            return st

        st = self._retry(_created_inner)
        self._last_activity = time.monotonic()
        return st.ctime

    # ── File open ────────────────────────────────────────────────────────

    def _open(
        self,
        path: str,
        mode: str = "rb",
        **kwargs: Any,
    ) -> DFSFile:
        return DFSFile(self, path, mode, **kwargs)

    # ── touch / pipe ─────────────────────────────────────────────────────

    def touch(self, path: str, truncate: bool = True, **kwargs: Any) -> None:
        """Create an empty file or update its modification time.

        Note: O_APPEND is not supported by DAOS DFS.
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "touch")
        if truncate:
            # Create or truncate — open with O_CREAT|O_TRUNC handles both
            with self.open(path, "wb"):
                pass
        else:
            # Try to update mtime on existing file; create if missing
            try:
                with self._lock:
                    obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDWR)
                    try:
                        now = time.time()
                        B.dfs_setattr(self._dfs, obj, mtime=now, atime=now)
                    finally:
                        B.dfs_release(obj)
            except FileNotFoundError:
                with self.open(path, "wb"):
                    pass

    def pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        """Write bytes to a file in one shot (no buffering overhead).

        Retried through ``_retry`` on transient errors. Path resolution happens
        inside the retried closure so a reconnect re-resolves against the fresh
        mount, and the write is idempotent (``O_CREAT | O_TRUNC``), so re-running
        after a transient failure cannot corrupt the target.
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "write")

        def _inner() -> None:
            parent, name = self._resolve_path(path)
            try:
                with self._lock:
                    obj = B.dfs_open_file(
                        self._dfs,
                        parent,
                        name,
                        stat_mod.S_IFREG | 0o644,
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                        oclass_id=self._object_class,
                        chunk_size=self._chunk_size,
                    )
                    try:
                        B.dfs_write(self._dfs, obj, 0, value)
                    finally:
                        B.dfs_close(obj)
            finally:
                self._release_parent(parent)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    # ── Symlink operations ───────────────────────────────────────────────

    def symlink(self, path: str, target: str, **kwargs: Any) -> None:
        """Create a symbolic link at `path` pointing to `target`."""
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "symlink")
        parent, name = self._resolve_path(path)
        try:
            with self._lock:
                obj = B.dfs_create_symlink(self._dfs, parent, name, target)
                B.dfs_release(obj)
        finally:
            self._release_parent(parent)

    def readlink(self, path: str) -> str:
        """Read the target of a symbolic link."""
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "readlink")

        def _readlink_inner() -> str:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                try:
                    return B.dfs_read_symlink(obj)
                finally:
                    B.dfs_release(obj)

        result = self._retry(_readlink_inner)
        self._last_activity = time.monotonic()
        return result

    def islink(self, path: str) -> bool:
        """Return True if path is a symbolic link."""
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _islink_inner() -> bool:
            with self._lock:
                obj, st = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                B.dfs_release(obj)
            return st.is_link

        try:
            result = self._retry(_islink_inner)
            self._last_activity = time.monotonic()
            return result
        except FileNotFoundError:
            return False
        except B.DaosError as exc:
            if exc.rc in (B.DER_NONEXIST, B.DER_NO_ENTRY):
                return False
            raise

    # ── Extended attributes ──────────────────────────────────────────────

    def getxattr(self, path: str, attr_name: str) -> bytes:
        """Get an extended attribute value from a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        attr_name : str
            Name of the extended attribute (e.g., "user.md5").

        Returns
        -------
        bytes
            The attribute value.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()

        def _inner() -> bytes:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                try:
                    return B.dfs_getxattr(self._dfs, obj, attr_name)
                finally:
                    B.dfs_release(obj)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def setxattr(self, path: str, attr_name: str, value: bytes, flags: int = 0) -> None:
        """Set an extended attribute on a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        attr_name : str
            Name of the extended attribute.
        value : bytes
            Attribute value.
        flags : int
            0 (default), B.XATTR_CREATE (fail if exists), or B.XATTR_REPLACE (fail if not exists).
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _inner() -> None:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                try:
                    B.dfs_setxattr(self._dfs, obj, attr_name, value, flags)
                finally:
                    B.dfs_release(obj)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    def listxattr(self, path: str) -> list[str]:
        """List all extended attribute names on a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.

        Returns
        -------
        list[str]
            List of attribute names.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()

        def _inner() -> list[str]:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                try:
                    return B.dfs_listxattr(self._dfs, obj)
                finally:
                    B.dfs_release(obj)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def removexattr(self, path: str, attr_name: str) -> None:
        """Remove an extended attribute from a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        attr_name : str
            Name of the extended attribute to remove.
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _inner() -> None:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, _O_NOFOLLOW)
                try:
                    B.dfs_removexattr(self._dfs, obj, attr_name)
                finally:
                    B.dfs_release(obj)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    # ── Access / permissions ─────────────────────────────────────────────

    def chmod(self, path: str, mode: int) -> None:
        """Change permissions on a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        mode : int
            New permission bits (e.g., 0o644).
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "chmod")
        parent, name = self._resolve_path(path)
        try:
            with self._lock:
                B.dfs_chmod(self._dfs, parent, name, mode)
        finally:
            self._release_parent(parent)

    # ── Pool space query ─────────────────────────────────────────────────

    def pool_space(self) -> B.PoolSpace:
        """Query the DAOS pool storage space.

        Returns
        -------
        B.PoolSpace
            Object with scm_total, scm_free, nvme_total, nvme_free (all in bytes).
        """
        self._check_closed()

        def _inner() -> B.PoolSpace:
            with self._lock:
                poh = B.dfs_pool_get(self._dfs)
                try:
                    return B.pool_query_space(poh)
                finally:
                    B.dfs_pool_put(self._dfs, poh)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def chown(self, path: str, uid: int, gid: int, follow_links: bool = True) -> None:
        """Change owner and group on a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        uid : int
            New owner UID (-1 to leave unchanged).
        gid : int
            New group GID (-1 to leave unchanged).
        follow_links : bool
            If False, operate on the symlink itself (O_NOFOLLOW).
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "chown")
        parent, name = self._resolve_path(path)
        flags = 0 if follow_links else _O_NOFOLLOW
        try:
            with self._lock:
                B.dfs_chown(self._dfs, parent, name, uid, gid, flags)
        finally:
            self._release_parent(parent)

    def access(self, path: str, mode: int = 0) -> bool:
        """Check access permissions on a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        mode : int
            Access mode to check. Use os.R_OK, os.W_OK, os.X_OK, os.F_OK (default).
            On DAOS, X_OK always returns True for directories.

        Returns
        -------
        bool
            True if access is granted, False otherwise.
        """
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"
        if path == "/":
            return True

        def _inner() -> bool:
            parent, name = self._resolve_path(path)
            try:
                with self._lock:
                    return B.dfs_access(self._dfs, parent, name, mode)
            finally:
                self._release_parent(parent)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def mount_attr(self) -> B.MountAttr:
        """Query the DFS mount attributes (default oclass, chunk_size, etc.).

        Returns
        -------
        B.MountAttr
            Object with oclass_id, dir_oclass_id, file_oclass_id, chunk_size, mode.
        """
        self._check_closed()

        def _inner() -> B.MountAttr:
            with self._lock:
                return B.dfs_query_attr(self._dfs)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def obj_info(self, path: str) -> B.ObjInfo:
        """Query object info (oclass, chunk_size, OID) for a file.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        B.ObjInfo
            Object with oclass_id, dir_oclass_id, file_oclass_id, chunk_size, oid_hi, oid_lo.
        """
        path = self._strip_protocol(path).rstrip("/")
        self._check_closed()

        def _inner() -> B.ObjInfo:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                try:
                    return B.dfs_obj_get_info(self._dfs, obj)
                finally:
                    B.dfs_release(obj)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def suggest_oclass(self, hint: str) -> int:
        """Suggest a DAOS object class from a hint string.

        Parameters
        ----------
        hint : str
            Hint string such as "file:single", "file:max", "dir:single", "dir:max".

        Returns
        -------
        int
            Object class ID suitable for use as ``object_class`` parameter.
        """
        self._check_closed()
        if not B._HAS_SUGGEST_OCLASS:
            raise NotImplementedError("dfs_suggest_oclass is not available in this DAOS build")

        def _inner() -> int:
            with self._lock:
                return B.dfs_suggest_oclass(self._dfs, hint)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    def set_prefix(self, prefix: str) -> None:
        """Set a prefix to trim from all DFS paths.

        This is a DFS mount-level setting: after calling this, all paths
        passed to DFS operations will have the prefix removed automatically.

        Parameters
        ----------
        prefix : str
            Path prefix to trim (e.g., "/my/container/root").
        """
        self._check_closed()
        if not B._HAS_SET_PREFIX:
            raise NotImplementedError("dfs_set_prefix is not available in this DAOS build")

        def _inner() -> None:
            with self._lock:
                B.dfs_set_prefix(self._dfs, prefix)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    # ── Directory operations ─────────────────────────────────────────────

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        if not path:
            return

        parts = [p for p in path.split("/") if p]
        if create_parents:
            if len(parts) > _DEFAULT_MAX_MKDIR_DEPTH:
                raise ValueError(
                    f"mkdir depth {len(parts)} exceeds max {_DEFAULT_MAX_MKDIR_DEPTH}. "
                    "Increase max_depth or create directories in smaller batches."
                )
            # Walk and create each level
            current = self._root
            intermediates = []
            try:
                for part in parts:
                    with self._lock:
                        try:
                            child, st = B.dfs_lookup_rel_path(self._dfs, current, part, os.O_RDWR)
                            if not st.is_dir:
                                raise FileExistsError(f"Not a directory: {part}")
                        except (FileNotFoundError, B.DaosError) as exc:
                            # Only create if the entry doesn't exist
                            if isinstance(exc, B.DaosError) and exc.rc not in (
                                B.DER_NONEXIST,
                                B.DER_NO_ENTRY,
                            ):
                                raise
                            B.dfs_mkdir(self._dfs, current, part)
                            child, _ = B.dfs_lookup_rel_path(self._dfs, current, part, os.O_RDWR)
                    intermediates.append(child)
                    current = child
            finally:
                # Serialize the releases against other libdfs calls on this
                # mount — handles are not thread-safe. Every other release site
                # holds the lock; this cleanup must too (the per-part loop above
                # already locks each C call individually).
                with self._lock:
                    for obj in intermediates:
                        B.dfs_release(obj)
        else:
            parent, name = self._resolve_path(path)
            try:
                with self._lock:
                    B.dfs_mkdir(self._dfs, parent, name)
            finally:
                self._release_parent(parent)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        if not path:
            return
        # mkdir(create_parents=True) is idempotent on existing directories, so
        # without this check makedirs(exist_ok=False) would silently succeed on
        # an existing path. fsspec/os semantics require FileExistsError instead.
        if not exist_ok and self.exists(path):
            raise FileExistsError(path)
        try:
            self.mkdir(path, create_parents=True)
        except FileExistsError:
            if exist_ok:
                return
            raise

    def rmdir(self, path: str) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "remove")
        parent, name = self._resolve_path(path)

        def _rmdir_inner() -> None:
            with self._lock:
                B.dfs_remove(self._dfs, parent, name, force=False)

        try:
            self._retry(_rmdir_inner)
        finally:
            self._release_parent(parent)

    def rm(self, path: str, recursive: bool = False, maxdepth: int | None = None) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        if maxdepth is not None:
            raise NotImplementedError("rm(maxdepth=...) is not supported by DAOS DFS")
        paths = path if isinstance(path, list) else [path]
        for p in paths:
            p = self._strip_protocol(p).rstrip("/")
            self._check_not_root(p or "/", "remove")
            parent, name = self._resolve_path(p)

            def _rm_inner() -> None:
                with self._lock:
                    B.dfs_remove(self._dfs, parent, name, force=recursive)

            try:
                self._retry(_rm_inner)
            finally:
                self._release_parent(parent)

    # ── Move / rename ────────────────────────────────────────────────────

    def mv(self, path1: str, path2: str, **kwargs: Any) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("Cannot move files in a read-only filesystem")
        path1 = self._strip_protocol(path1).rstrip("/")
        path2 = self._strip_protocol(path2).rstrip("/")
        self._check_not_root(path1 or "/", "move")
        self._check_not_root(path2 or "/", "move to")
        src_parent, src_name = self._resolve_path(path1)
        dst_parent, dst_name = self._resolve_path(path2)
        try:
            self._retry(self._mv_locked, src_parent, src_name, dst_parent, dst_name)
        finally:
            self._release_parent(src_parent)
            self._release_parent(dst_parent)

    def _mv_locked(self, src_parent: Any, src_name: str, dst_parent: Any, dst_name: str) -> None:
        with self._lock:
            B.dfs_move(self._dfs, src_parent, src_name, dst_parent, dst_name)

    # ── Walk (optimized — uses readdirplus, avoids extra stat) ───────────

    def walk(self, top: str, maxdepth: int | None = None, topdown: bool = True, **kwargs: Any):
        """Walk directory tree. Uses readdirplus for efficient stat batching.

        Parameters
        ----------
        top : str
            Root path to start walking from.
        maxdepth : int | None
            Maximum depth to descend (default: None = unlimited).
        topdown : bool
            If True, yield directory before its children (default: True).

        Note: O_APPEND is not supported by DAOS DFS.
        """
        self._check_closed()
        top = self._strip_protocol(top).rstrip("/") or "/"
        return self._walk_impl(top, maxdepth, topdown, depth=0)

    def _walk_impl(self, path: str, maxdepth: int | None, topdown: bool, depth: int):
        def _walk_inner() -> list[B.DirEntry]:
            with self._lock:
                if path == "/":
                    obj = self._root
                    need_release = False
                else:
                    obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                    need_release = True

                try:
                    entries = B.dfs_readdirplus(self._dfs, obj)
                finally:
                    if need_release:
                        B.dfs_release(obj)
            return entries

        entries = self._retry(_walk_inner)
        self._last_activity = time.monotonic()

        dirs = []
        files = []
        for entry in entries:
            if entry.stat and entry.stat.is_dir:
                dirs.append(entry.name)
            else:
                files.append(entry.name)

        if topdown:
            yield path, dirs, files

        if maxdepth is None or depth < maxdepth - 1:
            for d in dirs:
                child_path = path.rstrip("/") + "/" + d
                if child_path.startswith("//"):
                    child_path = child_path[1:]
                yield from self._walk_impl(child_path, maxdepth, topdown, depth + 1)

        if not topdown:
            yield path, dirs, files

    # ── Optimized bulk reads ─────────────────────────────────────────────

    def cat_file(self, path, start=None, end=None, **kwargs):
        """Read a single file, optionally byte range.

        Uses a single dfs_lookup_path call instead of the base class
        open→DFSFile→_resolve_path chain (3 C calls vs 7 for depth-3 paths).
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()

        def _inner() -> bytes:
            with self._lock:
                obj, st = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                try:
                    file_size = st.size
                    actual_start = (
                        start if start is None or start >= 0 else max(0, file_size + start)
                    )
                    actual_end = end if end is None or end >= 0 else max(0, file_size + end)
                    # Default values when start/end is None
                    if actual_start is None:
                        actual_start = 0
                    if actual_end is None:
                        actual_end = file_size
                    read_size = max(0, actual_end - actual_start)
                    if read_size == 0:
                        return b""
                    return B.dfs_read(self._dfs, obj, actual_start, read_size)
                finally:
                    B.dfs_release(obj)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    _CAT_RANGES_MAX_WORKERS = 8

    @property
    def _executor(self):
        """Lazy ThreadPoolExecutor for cat_ranges (thread-safe init)."""
        if self._cat_ranges_executor is None:
            with self._executor_lock:
                if self._cat_ranges_executor is None:
                    import concurrent.futures

                    self._cat_ranges_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self._CAT_RANGES_MAX_WORKERS
                    )
        return self._cat_ranges_executor

    def cat_ranges(self, paths, starts, ends, max_gap=None, on_error="return", **kwargs):
        """Read byte ranges from multiple files in parallel.

        Uses a ThreadPoolExecutor so each file's cat_file runs independently,
        serializing only on the per-call libdfs lock rather than reading
        sequentially.
        """
        self._check_closed()
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

        def _read_one(p, s, e):
            try:
                return self.cat_file(p, s, e)
            except Exception as exc:
                if on_error == "return":
                    return exc
                raise

        workers = min(n, self._CAT_RANGES_MAX_WORKERS)
        if n <= 1 or workers <= 1:
            return [_read_one(p, s, e) for p, s, e in zip(paths, starts, ends)]

        results = list(self._executor.map(lambda args: _read_one(*args), zip(paths, starts, ends)))
        return results

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        """Recursively list all files under *path*.

        Uses a single readdirplus call per directory (with batch stat),
        avoiding the walk→ls→info chain the base class uses.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()
        out = {}

        def _entry_info(name, entry, type_hint):
            st = entry.stat
            if st is None:
                return {"name": name, "size": 0, "type": type_hint}
            return {
                "name": name,
                "size": st.size if type_hint == "file" else 0,
                "type": type_hint,
                "mode": st.mode,
                "mtime": st.mtime,
                "atime": st.atime,
                "uid": st.uid,
                "gid": st.gid,
            }

        def _readdir(dir_path, depth):
            def _inner() -> list[B.DirEntry]:
                with self._lock:
                    if dir_path == "/":
                        obj = self._root
                        need_release = False
                    else:
                        obj, _ = B.dfs_lookup_path(self._dfs, dir_path, os.O_RDONLY)
                        need_release = True
                    try:
                        return B.dfs_readdirplus(self._dfs, obj)
                    finally:
                        if need_release:
                            B.dfs_release(obj)

            entries = self._retry(_inner)
            self._last_activity = time.monotonic()

            for entry in entries:
                full_path = dir_path.rstrip("/") + "/" + entry.name
                if full_path.startswith("//"):
                    full_path = full_path[1:]

                is_dir = entry.stat and entry.stat.is_dir
                if is_dir:
                    if withdirs:
                        out[full_path] = (
                            _entry_info(full_path, entry, "directory") if detail else {}
                        )
                    if maxdepth is None or depth < maxdepth - 1:
                        _readdir(full_path, depth + 1)
                else:
                    out[full_path] = _entry_info(full_path, entry, "file") if detail else {}

        # If path is a file (not a directory), return just that file
        try:
            info = self.info(path)
            if info.get("type") != "directory":
                if detail:
                    return {path: info}
                return [path]
        except FileNotFoundError:
            return [] if not detail else {}

        # Include the root dir itself if withdirs
        if withdirs:
            out[path] = self.info(path) if detail else {}

        _readdir(path, 0)

        names = sorted(out)
        if not detail:
            return names
        return {name: out[name] for name in names}

    # ── Optimized head / tail ─────────────────────────────────────────────

    def head(self, path: str, size: int = 1024) -> bytes:
        """Read the first ``size`` bytes of a file.

        Uses the optimized cat_file path with a byte range.
        """
        if size < 0:
            raise ValueError("size must be >= 0")
        return self.cat_file(path, start=0, end=size)  # type: ignore[no-any-return]

    def tail(self, path: str, size: int = 1024) -> bytes:
        """Read the last ``size`` bytes of a file.

        Uses the optimized cat_file path with a negative start offset.
        """
        if size < 0:
            raise ValueError("size must be >= 0")
        return self.cat_file(path, start=-size)  # type: ignore[no-any-return]

    def truncate(self, path: str, size: int = 0) -> None:
        """Truncate a file to the given size.

        Uses dfs_punch with DFS_MAX_FSIZE to truncate to the specified offset.
        Only available on DAOS versions that support dfs_punch.

        Parameters
        ----------
        path : str
            Path to the file.
        size : int
            New file size in bytes. Everything after this offset is removed.
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        if not B._HAS_PUNCH:
            raise NotImplementedError("dfs_punch is not available in this DAOS build")
        if size < 0:
            raise ValueError("size must be >= 0")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "truncate")

        def _inner() -> None:
            with self._lock:
                obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDWR)
                try:
                    B.dfs_punch(self._dfs, obj, size, B.DFS_MAX_FSIZE)
                finally:
                    B.dfs_release(obj)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    def find_parallel(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> list[str] | dict[str, Any]:
        """Recursively list all files under *path* using parallel directory iteration.

        Uses dfs_obj_anchor_split to partition large directories across
        multiple threads, each reading its own hash-space partition.

        A single ``ThreadPoolExecutor`` is created for the entire walk
        (not one per directory), avoiding unnecessary thread-pool
        creation/destruction overhead.

        Parameters
        ----------
        path : str
            Root directory to search.
        maxdepth : int | None
            Maximum depth to descend (default: None = unlimited).
        withdirs : bool
            If True, include directory entries in results.
        detail : bool
            If True, return dict of {path: info} instead of list of paths.
        num_workers : int
            Number of parallel workers for directory iteration (default: 4).
        """
        import concurrent.futures

        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()
        out: dict[str, Any] = {}

        def _entry_info(name: str, entry: B.DirEntry, type_hint: str) -> dict[str, Any]:
            st = entry.stat
            if st is None:
                return {"name": name, "size": 0, "type": type_hint}
            return {
                "name": name,
                "size": st.size if type_hint == "file" else 0,
                "type": type_hint,
                "mode": st.mode,
                "mtime": st.mtime,
                "atime": st.atime,
                "uid": st.uid,
                "gid": st.gid,
            }

        def _scan_partition(dir_path: str, anchor: Any, obj_handle: Any, result_list: list) -> None:
            """Read all entries in one anchor partition of a directory."""
            while not B._anchor_is_eof(anchor):
                # Serialize access to the DFS object handle to prevent
                # concurrent reads from the same handle (not thread-safe).
                with self._lock:
                    batch = B.dfs_readdirplus_at(self._dfs, obj_handle, anchor)
                if not batch:
                    break
                for entry in batch:
                    full_path = dir_path.rstrip("/") + "/" + entry.name
                    if full_path.startswith("//"):
                        full_path = full_path[1:]
                    is_dir = entry.stat and entry.stat.is_dir
                    result_list.append((full_path, is_dir, entry))

        def _readdir_seq(dir_path: str) -> list:
            """Sequential readdirplus for small directories or fallback."""
            with self._lock:
                if dir_path == "/":
                    obj = self._root
                    need_release = False
                else:
                    obj, _ = B.dfs_lookup_path(self._dfs, dir_path, os.O_RDONLY)
                    need_release = True
                try:
                    entries = B.dfs_readdirplus(self._dfs, obj)
                finally:
                    if need_release:
                        B.dfs_release(obj)
            result = []
            for entry in entries:
                full_path = dir_path.rstrip("/") + "/" + entry.name
                if full_path.startswith("//"):
                    full_path = full_path[1:]
                is_dir = entry.stat and entry.stat.is_dir
                result.append((full_path, is_dir, entry))
            return result

        def _readdir_parallel(
            dir_path: str, executor: concurrent.futures.ThreadPoolExecutor
        ) -> list | None:
            """Parallel readdirplus using anchor split with shared executor."""
            with self._lock:
                if dir_path == "/":
                    obj = self._root
                    need_release = False
                else:
                    obj, _ = B.dfs_lookup_path(self._dfs, dir_path, os.O_RDONLY)
                    need_release = True

            try:
                anchors = B.dfs_obj_anchor_split(obj, num_workers)
            except (B.DaosError, OSError, AttributeError, NotImplementedError):
                # anchor_split not supported — fall back to sequential
                if need_release:
                    with self._lock:
                        B.dfs_release(obj)
                return None

            if not anchors or len(anchors) <= 1:
                if need_release:
                    with self._lock:
                        B.dfs_release(obj)
                return None

            # Each worker reads its partition on the same obj handle
            worker_results: list[list] = [[] for _ in anchors]
            futures = []
            for i, anchor in enumerate(anchors):
                futures.append(
                    executor.submit(_scan_partition, dir_path, anchor, obj, worker_results[i])
                )
            concurrent.futures.wait(futures)

            # Release obj after all workers are done
            if need_release:
                with self._lock:
                    B.dfs_release(obj)

            # Flatten
            flat = []
            for w in worker_results:
                flat.extend(w)
            return flat

        def _walk_dir(
            dir_path: str, depth: int, executor: concurrent.futures.ThreadPoolExecutor
        ) -> None:
            """Walk directory tree, using parallel readdir for large dirs."""
            # Try parallel first, fall back to sequential
            entries = _readdir_parallel(dir_path, executor)
            if entries is None:
                entries = _readdir_seq(dir_path)

            self._last_activity = time.monotonic()

            for full_path, is_dir, entry in entries:
                if is_dir:
                    if withdirs:
                        out[full_path] = (
                            _entry_info(full_path, entry, "directory") if detail else {}
                        )
                    if maxdepth is None or depth < maxdepth - 1:
                        _walk_dir(full_path, depth + 1, executor)
                else:
                    out[full_path] = _entry_info(full_path, entry, "file") if detail else {}

        # If path is a file, return just that file
        try:
            info = self.info(path)
            if info.get("type") != "directory":
                if detail:
                    return {path: info}
                return [path]
        except FileNotFoundError:
            return [] if not detail else {}

        if withdirs:
            out[path] = self.info(path) if detail else {}

        # Use a single executor for the entire walk — avoids per-directory
        # ThreadPoolExecutor creation/destruction overhead.
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            _walk_dir(path, 0, executor)

        names = sorted(out)
        if not detail:
            return names
        return {name: out[name] for name in names}

    # ── Checksum / du / streaming transfer ─────────────────────────────

    def checksum(self, path: str) -> str:
        """Return a checksum for the file at path.

        Uses the file's mode, size, and mtime as a lightweight change
        indicator. This is not a content hash — it changes when the file
        is recreated (truncate+write), but not on append (which DFS
        doesn't support anyway).

        Suitable for cache invalidation in HuggingFace Datasets and Dask.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()

        def _checksum_inner() -> str:
            with self._lock:
                obj, st = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                try:
                    return f"{st.mode}:{st.size}:{st.mtime:.6f}"
                finally:
                    B.dfs_release(obj)

        result = self._retry(_checksum_inner)
        self._last_activity = time.monotonic()
        return result

    def du(self, path: str, total: bool = False, maxdepth: int | None = None, **kwargs):
        """Disk usage of files under path.

        Uses readdirplus for efficient batch stat instead of the base class
        implementation which calls ls(detail=True) per directory.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()
        sizes: dict[str, int] = {}

        def _du_walk(dir_path: str, depth: int) -> None:
            def _inner() -> list[B.DirEntry]:
                with self._lock:
                    if dir_path == "/":
                        obj = self._root
                        need_release = False
                    else:
                        obj, _ = B.dfs_lookup_path(self._dfs, dir_path, os.O_RDONLY)
                        need_release = True
                    try:
                        return B.dfs_readdirplus(self._dfs, obj)
                    finally:
                        if need_release:
                            B.dfs_release(obj)

            entries = self._retry(_inner)
            self._last_activity = time.monotonic()

            for entry in entries:
                full_path = dir_path.rstrip("/") + "/" + entry.name
                if full_path.startswith("//"):
                    full_path = full_path[1:]
                if entry.stat and entry.stat.is_dir:
                    if maxdepth is None or depth < maxdepth - 1:
                        _du_walk(full_path, depth + 1)
                else:
                    sizes[full_path] = entry.stat.size if entry.stat else 0

        # Check if path is a file
        try:
            info = self.info(path)
            if info.get("type") != "directory":
                if total:
                    return info.get("size", 0)
                return {path: info.get("size", 0)}
        except FileNotFoundError:
            return 0 if total else {}

        _du_walk(path, 0)

        if total:
            return sum(sizes.values())
        return sizes

    def get_file(self, rpath: str, lpath: str, **kwargs):
        """Download a file from DFS to a local path.

        Streams in 16MB chunks to avoid buffering the entire file in memory.
        The lock is released between chunks so other operations can interleave.
        """
        self._check_closed()
        rpath = self._strip_protocol(rpath).rstrip("/")
        self._check_not_root(rpath or "/", "download")

        def _inner() -> None:
            with self._lock:
                obj, st = B.dfs_lookup_path(self._dfs, rpath, os.O_RDONLY)
            size = st.size
            try:
                # Truncate the local file on each attempt so a retry after a
                # transient error restarts the download cleanly from offset 0.
                with open(lpath, "wb") as f:
                    offset = 0
                    while offset < size:
                        chunk_size = min(B._CHUNK_SIZE, size - offset)
                        with self._lock:
                            data = B.dfs_read(self._dfs, obj, offset, chunk_size)
                        if not data:
                            # Empty read == EOF. A short (non-empty) read is not
                            # EOF, so keep going rather than truncate the download.
                            break
                        f.write(data)
                        offset += len(data)
            finally:
                with self._lock:
                    B.dfs_release(obj)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    def put_file(self, lpath: str, rpath: str, **kwargs):
        """Upload a local file to DFS.

        Streams in 16MB chunks to avoid buffering the entire file in memory.
        The lock is released between chunks so other operations can interleave.
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        rpath = self._strip_protocol(rpath).rstrip("/")
        self._check_not_root(rpath or "/", "upload")

        def _inner() -> None:
            parent, name = self._resolve_path(rpath)
            try:
                with self._lock:
                    obj = B.dfs_open_file(
                        self._dfs,
                        parent,
                        name,
                        stat_mod.S_IFREG | 0o644,
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                        oclass_id=self._object_class,
                        chunk_size=self._chunk_size,
                    )
                try:
                    # O_TRUNC on open means a retry re-streams from offset 0.
                    offset = 0
                    with open(lpath, "rb") as f:
                        while True:
                            chunk = f.read(B._CHUNK_SIZE)
                            if not chunk:
                                break
                            with self._lock:
                                B.dfs_write(self._dfs, obj, offset, chunk)
                            offset += len(chunk)
                finally:
                    with self._lock:
                        B.dfs_close(obj)
            finally:
                self._release_parent(parent)

        self._retry(_inner)
        self._last_activity = time.monotonic()

    # ── Teardown ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Tear down DFS mount and DAOS connections.

        Shutdown order matters:
        1. Set ``self._closed = True`` so other threads fail fast.
        2. Shut down the ``_cat_ranges_executor`` (wait for in-flight tasks).
        3. Release root handle (must happen before umount/disconnect).
        4. Disconnect from DAOS (umount/cont_close/pool_disconnect).
        5. Clear all handle attributes.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            _logger.debug(
                "Closing DAOS connection pool=%s container=%s",
                self._pool_label,
                self._cont_label,
            )
            # Shut down the cached executor — wait for in-flight tasks
            if self._cat_ranges_executor is not None:
                self._cat_ranges_executor.shutdown(wait=True)
                self._cat_ranges_executor = None

            # Release root handle first (before umount/disconnect)
            if getattr(self, "_root", None) is not None:
                try:
                    B.dfs_release(self._root)
                except Exception:
                    _logger.warning("close(dfs_release) failed", exc_info=True)

            # Use dfs_disconnect for dfs_connect path, otherwise low-level teardown
            if getattr(self, "_uses_dfs_connect", False) and B._HAS_DFS_CONNECT:
                try:
                    _, libdfs = B._ensure_libs()
                    libdfs.dfs_disconnect(self._dfs)
                except Exception:
                    _logger.warning("close(dfs_disconnect) failed", exc_info=True)
            else:
                for name, attr in [
                    ("dfs_umount", "_dfs"),
                    ("cont_close", "_coh"),
                    ("pool_disconnect", "_poh"),
                ]:
                    handle = getattr(self, attr, None)
                    if handle is not None:
                        try:
                            if name == "dfs_umount":
                                B.dfs_umount(handle)
                            elif name == "cont_close":
                                B.cont_close(handle)
                            elif name == "pool_disconnect":
                                B.pool_disconnect(handle)
                        except Exception:
                            _logger.warning("close(%s) failed", name, exc_info=True)

            self._root = None
            self._dfs = None
            self._coh = None
            self._poh = None

    def __del__(self) -> None:
        # During interpreter shutdown / GC, the _bindings module may already
        # be garbage collected and DAOS C calls can segfault.
        # Users MUST call close() explicitly (or use a context manager).
        # __del__ only marks the object as closed without touching C handles.
        # This is acceptable: the handles leak until process exit.
        try:
            self._closed = True
        except Exception:
            pass
