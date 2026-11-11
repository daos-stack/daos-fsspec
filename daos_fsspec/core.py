# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""DFSFileSystem: fsspec filesystem for DAOS DFS containers.

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
import errno
import logging
import os
import stat as stat_mod
import tempfile
import threading
import time
import uuid
import weakref
from typing import Any, Callable, TypeVar

from fsspec.spec import AbstractFileSystem

from daos_fsspec import _bindings as B
from daos_fsspec import _fork

# same_handle is imported directly, not reached through `B`. It is pure
# attribute comparison with no libdaos dependency, so it never needs mocking --
# and routing it through the patched `B` would be actively harmful: a MagicMock
# returns a truthy Mock for any call, so `if B.same_handle(...): return` would
# skip every handle release in every mocked test. Three tests caught exactly
# that. Mocking exists here to avoid dlopen; this function does not dlopen.
from daos_fsspec._bindings import same_handle
from daos_fsspec._listing import DFSListingMixin
from daos_fsspec._rwlock import RWLock
from daos_fsspec.dfs_file import DFSFile

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# DFS dfs_lookup with O_RDONLY/O_RDWR returns EINVAL on symlinks.
# O_NOFOLLOW returns the symlink object itself without following.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0x20000)

# Default retry configuration
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_BACKOFF = 0.1  # seconds

# Listings-cache TTL, used only when caching is explicitly enabled; it is off
# by default, see the class docstring. Mirrors dfuse's dentry-time default of
# about one second. This is a staleness budget, not a performance knob: a peer
# rank writing into a shared directory cannot invalidate our cache, so the only
# guarantee available is that we notice within this long.
_DEFAULT_LISTINGS_EXPIRY = 1.0  # seconds

# DFS limits (from daos_fs.h)
DFS_MAX_PATH = 4096
DFS_MAX_NAME = 255
_DEFAULT_MAX_MKDIR_DEPTH = 100


class MountChangedError(OSError):
    """A reconnect replaced the mount an in-flight operation was using.

    Raised by multi-chunk reads and writes, which deliberately release the
    lock between chunks so other operations can interleave. That window is
    real: another thread hitting a retryable error in, say, ``ls()`` sends
    ``_retry`` into ``_reconnect()``, which umounts the filesystem and mounts
    a fresh one. The object handle the in-flight operation holds then belongs
    to a mount that no longer exists, and handing it to libdfs is a
    use-after-free rather than an error.

    ``_mount_generation`` already existed to catch exactly this, but only
    teardown paths consulted it; the read and write paths did not, which is
    the gap this closes.

    Carries ``errno.EAGAIN``, so ``_is_retryable`` treats it as transient and
    ``_retry`` restarts the operation. That restart is correct: every retried
    callable re-resolves its path, so the second attempt runs entirely on the
    new mount. An operation on an already-open ``DFSFile`` is not wrapped in
    ``_retry`` and so surfaces this to the caller, which is also correct --
    that file's handle cannot be revived.
    """

    def __init__(self, operation: str):
        super().__init__(
            errno.EAGAIN,
            f"the DAOS mount was replaced by a reconnect while {operation} was "
            "in flight; its handles belong to the previous mount. Retry the "
            "operation.",
        )


class DFSFileSystem(DFSListingMixin, AbstractFileSystem):
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
    use_listings_cache : bool
        Cache ``ls()`` results (default: **False**). See "Listings cache".
    listings_expiry_time : float
        Seconds a cached listing stays valid when caching is enabled
        (default: 1.0, mirroring dfuse's ``dentry-time``).

    Listings cache
    --------------
    Disabled by default. Only ls() consults this cache. Local mutations invalidate
    cached entries; peer processes cannot invalidate them. Use a bounded TTL or
    invalidate_cache() after coordination with other writers."""

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
        # libdaos cannot be reinitialized after fork(): its process-global
        # initialized flag and orphaned background-thread locks are inherited.
        _fork._check_can_initialize()
        # Listings cache: OFF by default. Opt in with use_listings_cache=True.
        # Peer processes cannot invalidate local cached listings.
        kwargs.setdefault("use_listings_cache", False)
        # The TTL still has to be set explicitly for anyone who does opt in.
        # fsspec's DirCache defaults listings_expiry_time to None, meaning
        # entries never expire; unusable here, because a peer rank creating a
        # file in a shared directory cannot invalidate our cache. A bounded
        # staleness window is the whole safety argument.
        kwargs.setdefault("listings_expiry_time", _DEFAULT_LISTINGS_EXPIRY)
        super().__init__(**kwargs)
        self._pool_label = pool
        self._cont_label = container
        self._readonly = readonly
        self._object_class = object_class
        self._chunk_size = chunk_size
        # Shared/exclusive rather than mutual exclusion: DFS does its own
        # locking for mount state (see _rwlock.py for the source citations),
        # so concurrent I/O needs no serialization. The only invariant we
        # enforce is that teardown waits for in-flight libdfs calls to drain --
        # calls, not whole operations: chunked I/O releases the lock between
        # chunks, which is why those paths also check _mount_generation.
        # Reentrant, so helpers that release handles (e.g. _release_parent,
        # _resolve_path cleanup) work from both locked and unlocked contexts.
        self._closed = False
        self._lock = RWLock(read_guard=self._check_closed)

        # Fork tracking. DAOS handles are unusable in a forked child; we
        # detect that and raise instead of hanging. See _fork.py.
        self._pid = os.getpid()
        self._fork_generation = _fork.generation()
        self._forked = False

        # Bumped by every _connect(); see there. Handles record the generation
        # they came from so a stale one can be recognised rather than used.
        self._mount_generation = 0

        # Set while a thread is executing a cat_ranges task on _executor. A
        # read that is already running there must not pipeline: it would
        # submit sub-reads into the very pool it occupies and block on them.
        # See _read_pipelined.
        self._on_read_pool = threading.local()

        # The bindings module these handles will belong to. Checked by the
        # finaliser: a handle may only be torn down through the same library
        # that produced it. In production this never changes, so the check is
        # free. It matters when the module has been swapped; a reload, or a
        # test that patched the ctypes layer; because the finaliser may then
        # run after the swap, and handing a handle to a different library (or
        # to a mock, which allocates during collection) is a segfault rather
        # than an error. This is the same hazard that took down the live suite
        # once as a leaked mock and once as this.
        self._bindings: Any = B

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

        # Instance calls must validate authority before fsspec strips the URI.
        self._strip_protocol = self._strip_bound_protocol  # type: ignore[method-assign]

        # Connect to DAOS
        self._uses_dfs_connect = False
        self._open_files: dict[int, weakref.ReferenceType[Any]] = {}
        self._cleanup_error: str | None = None
        self._connect()

        # Only register once we hold real handles: a failed connect has
        # nothing for the fork hook to poison.
        _fork.register(self)

    # ── Fork safety ──────────────────────────────────────────────────────

    def _poison_after_fork(self) -> None:
        """Mark handles unusable in a forked child. Called by the fork hook.

        Deliberately touches no DAOS state: the whole point is that calling
        into libdfs here would hang. The inherited handles leak until this
        child exits, which is harmless: the parent still owns the real
        connections and will tear them down properly.
        """
        self._forked = True
        # These may have been held by threads that do not exist in this child.
        self._lock = RWLock(read_guard=self._check_closed)
        self._executor_lock = threading.Lock()
        # Its worker threads did not survive the fork; drop the reference
        # without shutting it down (shutdown would join threads that are gone).
        self._cat_ranges_executor = None

    def _check_fork(self) -> None:
        """Raise if this filesystem is being used from a forked child."""
        if self._forked or self._fork_generation != _fork.generation():
            raise _fork.fork_error(self._pid)

    def _connect(self) -> None:
        with B._runtime.connection(self):
            try:
                self._connect_impl()
            except BaseException:
                self._closed = True
                try:
                    self._teardown_handles()
                except Exception:
                    _logger.warning("cleanup after failed connection failed", exc_info=True)
                raise

    def _connect_impl(self) -> None:
        """Establish DAOS connection.

        Uses dfs_connect() when available (single-call API, DAOS 2.6+),
        otherwise falls back to the 4-step: daos_init → pool_connect → cont_open → dfs_mount.
        """

        self._mount_generation += 1
        # Load the libraries before branching. B._HAS_DFS_CONNECT starts False
        # and is only set by _setup_signatures, which runs inside
        # _ensure_libs(); and nothing loads the libraries at import. So the
        # *first* mount in a process always read False and took the low-level
        # path, while every later mount took the dfs_connect path: two
        # different connect and teardown shapes, and different handles
        # populated (_poh/_coh set versus None), decided by construction order.
        # A _reconnect() of that first filesystem would then silently switch it
        # to the other path mid-life.
        B._ensure_libs()
        if B._HAS_DFS_CONNECT:
            self._connect_via_dfs_connect()
        else:
            self._connect_via_lowlevel()

    def _connect_via_dfs_connect(self) -> None:
        """Connect using the high-level dfs_connect() API (one call: pool+cont+mount)."""
        _, libdfs = B._ensure_libs()
        B.daos_init()
        # dfs_connect() requires dfs_init() to have been called
        B._dfs_init()
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
        self._root, _ = B.dfs_lookup_path(
            self._dfs, "/", os.O_RDONLY if self._readonly else os.O_RDWR
        )
        self._last_activity = time.monotonic()
        _logger.debug(
            "Connected via dfs_connect pool=%s container=%s", self._pool_label, self._cont_label
        )

    def _connect_via_lowlevel(self) -> None:
        """Connect using the low-level 4-step API (legacy path)."""
        self._uses_dfs_connect = False
        B.daos_init()
        self._poh = B.pool_connect(
            self._pool_label, flags=B.DAOS_PC_RO if self._readonly else B.DAOS_PC_RW
        )
        self._coh = B.cont_open(
            self._poh,
            self._cont_label,
            flags=B.DAOS_COO_RO if self._readonly else B.DAOS_COO_RW,
        )
        self._dfs = B.dfs_mount_fs(self._poh, self._coh, readonly=self._readonly)
        self._root, _ = B.dfs_lookup_path(
            self._dfs, "/", os.O_RDONLY if self._readonly else os.O_RDWR
        )
        self._last_activity = time.monotonic()
        _logger.debug(
            "Connected via lowlevel pool=%s container=%s", self._pool_label, self._cont_label
        )

    def _reconnect(self, *, automatic: bool = False) -> None:
        """Tear down and re-establish the DAOS connection.

        Sets ``self._closed = True`` during teardown so other threads see
        invalid handles immediately.  If ``_connect()`` fails, the
        filesystem remains closed: the caller will see a ``RuntimeError``
        on the next operation.
        """
        self._check_fork()
        _logger.warning(
            "Reconnecting to DAOS pool=%s container=%s", self._pool_label, self._cont_label
        )
        # Exclusive: tears down handles other threads may be mid-call on, so
        # it must wait for in-flight libdfs calls to drain. Operations spanning
        # several chunks can still be part-way through; they detect this mount
        # going away via _mount_generation.
        with self._lock.write():
            if automatic and self._open_files:
                raise RuntimeError("automatic reconnect refused while files are open")
            # Mark closed during teardown so other threads see it immediately
            self._closed = True

            self._teardown_handles()

            # Re-establish: stays closed if this fails
            self._connect()
            self._closed = False

    def _is_retryable(self, exc: Exception) -> bool:
        """Return True if the exception is a transient DAOS error."""
        # Intrinsic rather than inferred from its errno. MountChangedError does
        # carry EAGAIN, which B.RETRYABLE_ERRORS includes, but making
        # retryability depend on that coupling means any test or caller with
        # its own idea of the retryable set silently changes whether a
        # reconnect is recoverable.
        if isinstance(exc, MountChangedError):
            return True
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
                # Try reconnecting on connection-level errors.
                #
                # DER_NO_HDL used to be listed here and was dead code: it is not
                # in B.RETRYABLE_ERRORS, so the `not self._is_retryable(exc)`
                # check above re-raises before this point is ever reached.
                #
                # It is deliberately not made retryable. An invalid handle is
                # not a transient fabric condition; it means a handle was
                # released, closed, or inherited across a fork, which is a local
                # bug rather than something a retry can clear. More to the
                # point, _reconnect() rebuilds the *mount* and nothing else: it
                # cannot reopen the object handles held by live DFSFile objects
                # or resolved by a caller. Retrying after a "successful"
                # reconnect would hand those stale handles straight back to
                # libdfs, which is undefined behaviour rather than an error
                # code. Fork is already handled; _reconnect() calls
                # _check_fork() first; and a genuinely evicted pool handle is
                # better surfaced to the caller than silently papered over.
                if isinstance(exc, B.DaosError) and exc.rc in (
                    B.DER_UNREACH,
                    B.DER_HG,
                    B.DER_TIMEDOUT,
                ):
                    try:
                        self._reconnect(automatic=True)
                    except Exception as reconnect_exc:
                        _logger.error(
                            "Reconnect failed on attempt %d: %s",
                            attempt + 1,
                            reconnect_exc,
                        )

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
        if self._closed or self._forked or self._fork_generation != _fork.generation():
            return False
        try:
            with self._lock.read():
                B.dfs_stat(self._dfs, self._root)
            self._last_activity = time.monotonic()
            return True
        except Exception:
            return False

    @property
    def is_alive(self) -> bool:
        """Return True if the connection appears to be alive."""
        if self._forked or self._fork_generation != _fork.generation():
            return False
        return not self._closed

    def __repr__(self) -> str:
        return (
            f"DFSFileSystem(pool={self._pool_label!r}, "
            f"container={self._cont_label!r}, "
            f"readonly={self._readonly})"
        )

    def __reduce__(self):
        # Deliberately raises rather than reconnecting on unpickle. Pickling by
        # reconnect would make `self.fs = fs` on a torch Dataset appear to work
        # while silently opening a pool+container handle per worker; 24 ranks
        # x 4 workers is 96 connections. Failing loudly costs one documented
        # line and keeps that fan-out visible.
        raise TypeError(
            "DFSFileSystem cannot be pickled: DAOS handles do not survive "
            "process boundaries, and reconnecting on unpickle would silently "
            "open one pool+container handle per worker.\n"
            "\n"
            "Use daos_fsspec.get_fs(pool, container) instead. It returns a "
            "per-process cached filesystem, so keep only the pool and "
            "container strings on your object (strings pickle) and call "
            "get_fs() where you need I/O:\n"
            "\n"
            "    def __getitem__(self, i):\n"
            "        fs = daos_fsspec.get_fs(self.pool, self.container)\n"
            "        return fs.cat_file(self.files[i])\n"
            "\n"
            "Note that fork() is unsupported regardless; use spawn."
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

    def _strip_bound_protocol(self, path: str) -> str:
        if path.startswith("dfs://"):
            authority = type(self)._get_kwargs_from_urls(path)
            if authority != {"pool": self._pool_label, "container": self._cont_label}:
                raise ValueError("DFS URI pool/container does not match this filesystem")
        return type(self)._strip_protocol(path)

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
        """Raise if the filesystem is closed or inherited across a fork."""
        # Fork first: a forked child also looks "open", and the fork message
        # is far more actionable than "DFSFileSystem is closed".
        self._check_fork()
        if self._closed:
            raise RuntimeError("DFSFileSystem is closed")

    @staticmethod
    def _check_not_root(path: str, operation: str) -> None:
        """Raise if an operation that requires a directory entry targets root."""
        if path == "/":
            raise ValueError(f"cannot {operation} the DFS root")

    def _resolve_path(self, path: str) -> tuple[Any, str, int]:
        """Resolve a path into ``(parent_obj, basename, mount_generation)``.

        For "/", returns ``(root_obj, "", generation)``.
        For "/a/b/c", walks the tree to return
        ``(obj_for_/a/b, "c", generation)``.

        All intermediate handles are released; only the final parent
        is kept alive (caller must release if != root).
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        # Keep one shared hold for the walk. Besides making close admission
        # atomic, this prevents a reconnect from swapping the mount between two
        # components. The returned parent can outlive this scope, so its mount
        # generation travels with it and every caller must validate that tag.
        with self._lock.read():
            generation = self._mount_generation
            if path == "/":
                return self._root, "", generation

            parts = [p for p in path.split("/") if p]
            if len(parts) == 1:
                return self._root, parts[0], generation

            parent = self._root
            intermediates: list[Any] = []
            lookup_flags = os.O_RDONLY if self._readonly else os.O_RDWR
            try:
                for part in parts[:-1]:
                    child, _ = B.dfs_lookup_rel_path(self._dfs, parent, part, lookup_flags)
                    intermediates.append(child)
                    parent = child
            except Exception:
                for obj in intermediates:
                    if not same_handle(obj, self._root):
                        B.dfs_release(obj)
                raise

            # Release all intermediates except the final parent.
            for obj in intermediates[:-1]:
                if not same_handle(obj, self._root):
                    B.dfs_release(obj)

            return parent, parts[-1], generation

    def _assert_same_mount(self, generation: int, operation: str) -> None:
        """Raise if a reconnect replaced the mount since ``generation``.

        Call inside the lock, immediately before handing a handle to libdfs,
        on any path that releases the lock between calls.
        """
        if generation != self._mount_generation:
            raise MountChangedError(operation)

    def _release_parent(self, parent: Any, generation: int | None = None) -> None:
        """Release a parent handle if it's not the root."""
        if parent is None:
            return
        # A NULL handle refers to nothing; handing it to dfs_release is at best
        # a wasted error return. MagicMocks are truthy, so this does not change
        # what the mocked tests exercise.
        if getattr(parent, "value", 0) is None:
            return
        # Cleanup must be allowed to enter after close/reconnect so it can
        # inspect the generation, but it must never call libdfs in that case.
        with self._lock._read_unchecked():
            if self._closed or self._forked or self._fork_generation != _fork.generation():
                return
            if generation is not None and generation != self._mount_generation:
                return
            if same_handle(parent, self._root):
                return
            B.dfs_release(parent)

    def _release_object(self, obj: Any, generation: int, *, close: bool = False) -> None:
        """Release a current-mount object, or safely drop a stale reference."""
        if obj is None or getattr(obj, "value", 0) is None:
            return
        with self._lock._read_unchecked():
            if self._closed or self._forked or self._fork_generation != _fork.generation():
                return
            if generation != self._mount_generation:
                return
            if close:
                B.dfs_close(obj)
            else:
                B.dfs_release(obj)

    @staticmethod
    def _validate_path(path: str) -> str:
        """Validate and normalize a DFS path.

        Raises ValueError if the path is invalid.
        """
        if "\x00" in path:
            raise ValueError("path must not contain NUL bytes")
        if len(path.encode("utf-8")) > DFS_MAX_PATH:
            raise ValueError(
                f"path too long ({len(path.encode('utf-8'))} > {DFS_MAX_PATH}): {path[:80]}..."
            )
        # Validate individual components
        parts = path.split("/")
        for part in parts:
            if part and len(part.encode("utf-8")) > DFS_MAX_NAME:
                raise ValueError(
                    f"path component too long ({len(part.encode('utf-8'))} > {DFS_MAX_NAME}): {part[:80]}..."
                )
        return path

    # ── File info / stat ─────────────────────────────────────────────────

    def info(self, path: str, **kwargs: Any) -> dict:
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        def _info_inner() -> B.StatResult:
            with self._lock.read():
                if path == "/":
                    st = B.dfs_stat(self._dfs, self._root)
                else:
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
            with self._lock.read():
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
            with self._lock.read():
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
            # Create or truncate: open with O_CREAT|O_TRUNC handles both
            with self.open(path, "wb"):
                pass
        else:
            # Try to update mtime on existing file; create if missing
            try:
                with self._lock.read():
                    obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDWR)
                    try:
                        now = time.time()
                        B.dfs_setattr(self._dfs, obj, mtime=now, atime=now)
                    finally:
                        B.dfs_release(obj)
            except FileNotFoundError:
                with self.open(path, "wb"):
                    pass
        self._invalidate_listing(path)

    def pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        """Write bytes to a file in one shot (no buffering overhead)."""
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "write")
        parent, name, generation = self._resolve_path(path)
        try:
            with self._lock.read():
                self._assert_same_mount(generation, "writing a file")
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
            self._release_parent(parent, generation)
        self._invalidate_listing(path)

    # ── Symlink operations ───────────────────────────────────────────────

    def symlink(self, path: str, target: str, **kwargs: Any) -> None:
        """Create a symbolic link at `path` pointing to `target`."""
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "symlink")
        parent, name, generation = self._resolve_path(path)
        try:
            with self._lock.read():
                self._assert_same_mount(generation, "creating a symlink")
                obj = B.dfs_create_symlink(self._dfs, parent, name, target)
                B.dfs_release(obj)
        finally:
            self._release_parent(parent, generation)
        self._invalidate_listing(path)

    def readlink(self, path: str) -> str:
        """Read the target of a symbolic link."""
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/")
        self._check_not_root(path or "/", "readlink")

        def _readlink_inner() -> str:
            with self._lock.read():
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
            with self._lock.read():
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
            with self._lock.read():
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
            with self._lock.read():
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
            with self._lock.read():
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
            with self._lock.read():
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
        parent, name, generation = self._resolve_path(path)
        try:
            with self._lock.read():
                self._assert_same_mount(generation, "changing permissions")
                B.dfs_chmod(self._dfs, parent, name, mode)
        finally:
            self._release_parent(parent, generation)
        # "mode" is one of the fields ls() caches, so a chmod that skipped
        # this would leave the old permissions visible for up to the TTL.
        self._invalidate_listing(path)

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
            with self._lock.read():
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
        parent, name, generation = self._resolve_path(path)
        flags = 0 if follow_links else _O_NOFOLLOW
        try:
            with self._lock.read():
                self._assert_same_mount(generation, "changing ownership")
                B.dfs_chown(self._dfs, parent, name, uid, gid, flags)
        finally:
            self._release_parent(parent, generation)
        self._invalidate_listing(path)

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
            parent, name, generation = self._resolve_path(path)
            try:
                with self._lock.read():
                    self._assert_same_mount(generation, "checking access")
                    return B.dfs_access(self._dfs, parent, name, mode)
            finally:
                self._release_parent(parent, generation)

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
            with self._lock.read():
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
            with self._lock.read():
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
            with self._lock.read():
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
            with self._lock.read():
                B.dfs_set_prefix(self._dfs, prefix)

        self._retry(_inner)
        self._last_activity = time.monotonic()
        # Prefix changes reinterpret every cached key, not just one parent.
        self.invalidate_cache()

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

            with self._lock.read():
                current = self._root
                intermediates = []
                try:
                    for part in parts:
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
                            try:
                                B.dfs_mkdir(self._dfs, current, part)
                            except FileExistsError:
                                pass
                            child, _ = B.dfs_lookup_rel_path(self._dfs, current, part, os.O_RDWR)
                        intermediates.append(child)
                        current = child
                finally:
                    for obj in intermediates:
                        B.dfs_release(obj)
            # Every level walked may have been created, and each creation
            # changes its own parent's listing; so invalidate the whole
            # chain, not just the leaf.
            self._invalidate_listing(*("/" + "/".join(parts[: i + 1]) for i in range(len(parts))))
        else:
            parent, name, generation = self._resolve_path(path)
            try:
                with self._lock.read():
                    self._assert_same_mount(generation, "creating a directory")
                    B.dfs_mkdir(self._dfs, parent, name)
            finally:
                self._release_parent(parent, generation)
            self._invalidate_listing(path)

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

        def _rmdir_inner() -> None:
            parent, name, generation = self._resolve_path(path)
            try:
                with self._lock.read():
                    self._assert_same_mount(generation, "removing a directory")
                    B.dfs_remove(self._dfs, parent, name, force=False)
            finally:
                self._release_parent(parent, generation)

        self._retry(_rmdir_inner)
        self._invalidate_listing(path)

    def rm(self, path: str, recursive: bool = False, maxdepth: int | None = None) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        if maxdepth is not None:
            raise NotImplementedError("rm(maxdepth=...) is not supported by DAOS DFS")
        paths = path if isinstance(path, list) else [path]
        paths = [self._strip_protocol(p).rstrip("/") for p in paths]
        for p in paths:
            self._check_not_root(p or "/", "remove")

            def _rm_inner() -> None:
                parent, name, generation = self._resolve_path(p)
                try:
                    with self._lock.read():
                        self._assert_same_mount(generation, "removing a path")
                        B.dfs_remove(self._dfs, parent, name, force=recursive)
                finally:
                    self._release_parent(parent, generation)

            self._retry(_rm_inner)

            self._invalidate_listing(p)
            if recursive:
                # A recursive remove invalidates listings of everything beneath
                # p, whose paths we no longer know. Nothing cheaper than
                # dropping the lot is correct here.
                self.dircache.clear()

    # ── Move / rename ────────────────────────────────────────────────────

    def mv(self, path1: str, path2: str, **kwargs: Any) -> None:
        self._check_closed()
        if self._readonly:
            raise PermissionError("Cannot move files in a read-only filesystem")
        path1 = self._strip_protocol(path1).rstrip("/")
        path2 = self._strip_protocol(path2).rstrip("/")
        self._check_not_root(path1 or "/", "move")
        self._check_not_root(path2 or "/", "move to")

        def _mv_inner() -> None:
            # Resolved per attempt, inside the retried callable. This used to
            # resolve both parents once and pass the handles into _retry, which
            # is unsafe for one specific reason: _retry() calls _reconnect() on
            # DER_UNREACH / DER_HG / DER_TIMEDOUT, and _reconnect() umounts the
            # filesystem these handles belong to before mounting a fresh one.
            # The next attempt then handed dfs_move a handle from a mount that
            # no longer existed.
            #
            # The release was wrong in the same way. _release_parent() skips the
            # root by identity (`parent != self._root`), and after a reconnect
            # self._root is a *new* object; so the old root stopped matching
            # and got released again, on top of the release _reconnect() had
            # already done. For a depth-1 path like "/a", where _resolve_path
            # returns the root itself, that was dfs_release() three times on one
            # handle.
            #
            # Re-resolving per attempt fixes both: every handle now comes from
            # the mount that is current when it is used, and is released while
            # that mount is still the live one.
            src_parent, src_name, src_generation = self._resolve_path(path1)
            try:
                dst_parent, dst_name, dst_generation = self._resolve_path(path2)
                try:
                    self._mv_locked(
                        src_parent,
                        src_name,
                        src_generation,
                        dst_parent,
                        dst_name,
                        dst_generation,
                    )
                finally:
                    self._release_parent(dst_parent, dst_generation)
            finally:
                self._release_parent(src_parent, src_generation)

        self._retry(_mv_inner)
        self._invalidate_listing(path1, path2)

    def _mv_locked(
        self,
        src_parent: Any,
        src_name: str,
        src_generation: int,
        dst_parent: Any,
        dst_name: str,
        dst_generation: int,
    ) -> None:
        with self._lock.read():
            self._assert_same_mount(src_generation, "moving a path")
            self._assert_same_mount(dst_generation, "moving a path")
            B.dfs_move(self._dfs, src_parent, src_name, dst_parent, dst_name)

    # ── Optimized bulk reads ─────────────────────────────────────────────

    def cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        """Read a single file, optionally byte range.

        Uses a single dfs_lookup_path call instead of the base class
        open→DFSFile→_resolve_path chain (3 C calls vs 7 for depth-3 paths).
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()

        def _inner() -> bytes:
            # The lock is taken for the lookup and dropped before the read, not
            # held across it. _read_pipelined may wait on executor workers that
            # each take the shared lock themselves, and holding it while waiting
            # on them is the one shape that can deadlock against a waiting
            # writer; find() already follows this discipline for the same
            # reason. Every individual libdfs call is still made under the lock;
            # obj is a per-call handle and is never shared across threads other
            # than the workers reading this one range.
            with self._lock.read():
                generation = self._mount_generation
                obj, st = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
            try:
                file_size = st.size
                actual_start = start if start is None or start >= 0 else max(0, file_size + start)
                actual_end = end if end is None or end >= 0 else max(0, file_size + end)
                # Default values when start/end is None
                if actual_start is None:
                    actual_start = 0
                if actual_end is None:
                    actual_end = file_size
                read_size = max(0, actual_end - actual_start)
                if read_size == 0:
                    return b""
                return self._read_pipelined(obj, actual_start, read_size, generation=generation)
            finally:
                self._release_object(obj, generation)

        result = self._retry(_inner)
        self._last_activity = time.monotonic()
        return result

    _CAT_RANGES_MAX_WORKERS = 8

    _PIPELINE_MIN_BYTES = 4 * B.CHUNK_SIZE  # 64 MB

    def _read_pipelined(
        self,
        obj: Any,
        offset: int,
        size: int,
        *,
        generation: int | None = None,
    ) -> bytes:
        """Read ``size`` bytes at ``offset``, overlapping the chunk reads.

        ``B.dfs_read`` splits anything over ``CHUNK_SIZE`` into sequential
        calls. Issuing those concurrently is only safe-and-useful because of
        the lock narrowing: I/O takes the shared lock, and ctypes
        releases the GIL around the foreign call, so the reads genuinely
        overlap instead of queueing on a mutex.

        Short-read semantics are preserved exactly, and they are the whole
        subtlety here. ``dfs_read`` may return fewer bytes than asked for at a
        chunk or stripe boundary without being at EOF, so a short read is not
        an EOF signal; only an empty read is. Each worker therefore loops
        until its own sub-range is full or it reads nothing. A worker that
        comes back short has genuinely hit EOF inside its range, so everything
        after it is past EOF and is dropped; concatenating later pieces would
        fabricate data across a hole.
        """
        # Read serially when too small to benefit, and; critically; when
        # this thread is already running a cat_ranges task on _executor.
        #
        # _executor is shared: cat_ranges() runs one task per path on it, and
        # each task calls cat_file() -> here. Submitting chunk reads into the
        # same pool and blocking on them is a textbook thread-pool deadlock,
        # and with _CAT_RANGES_MAX_WORKERS == 8 it needed only 8 paths of
        # >= _PIPELINE_MIN_BYTES to hang permanently: every worker a blocked
        # parent, every child unschedulable. close() then hung too, in
        # executor.shutdown(wait=True), leaving a process that would not exit.
        #
        # Declining to pipeline here costs nothing real. cat_ranges already
        # provides parallelism across paths, which is the same libdfs
        # concurrency pipelining would buy; just spread over files instead
        # of over chunks of one file.
        # Production callers pass the generation captured atomically with the
        # object lookup. Keeping the default supports direct/internal callers,
        # but the sample is still taken under the lifecycle lock.
        if generation is None:
            with self._lock.read():
                generation = self._mount_generation

        if size < self._PIPELINE_MIN_BYTES or getattr(self._on_read_pool, "active", False):
            with self._lock.read():
                self._assert_same_mount(generation, "a read")
                single: bytes = B.dfs_read(self._dfs, obj, offset, size)
            return single

        chunk = B.CHUNK_SIZE
        bounds = [
            (off, min(chunk, offset + size - off)) for off in range(offset, offset + size, chunk)
        ]

        def _read_one(off: int, want: int) -> bytes:
            parts: list[bytes] = []
            pos, remaining = off, want
            while remaining > 0:
                with self._lock.read():
                    self._assert_same_mount(generation, "a pipelined read")
                    data: bytes = B.dfs_read(self._dfs, obj, pos, remaining)
                if not data:
                    break  # genuine EOF: empty read is the only such signal
                parts.append(data)
                pos += len(data)
                remaining -= len(data)
            return b"".join(parts)

        executor = self._executor
        futures = []
        try:
            # Submit explicitly rather than with a comprehension: close() may
            # begin executor shutdown between submissions. If submit then
            # raises, the futures already accepted still have to be drained
            # before the caller is allowed to release ``obj``.
            for off, want in bounds:
                futures.append(executor.submit(_read_one, off, want))

            pieces: list[bytes] = []
            for (_off, want), future in zip(bounds, futures):
                piece = future.result()
                pieces.append(piece)
                if len(piece) < want:
                    # EOF inside this chunk. Stop here: later chunks are past
                    # EOF, but they may already be running on this same handle.
                    break
            return b"".join(pieces)
        finally:
            # cancel() handles work that has not started; wait() is required
            # for workers already inside dfs_read. Without both, cat_file's
            # finally releases the object while those workers still use it.
            for future in futures:
                future.cancel()
            if futures:
                import concurrent.futures

                concurrent.futures.wait(futures)

    @property
    def _executor(self):
        """Lazy ThreadPoolExecutor for cat_ranges (thread-safe init)."""
        with self._executor_lock:
            # close() takes this same mutex while publishing _closed and
            # detaching the executor, so one created here is either captured by
            # close or creation fails cleanly; never leaked after teardown.
            self._check_closed()
            if self._cat_ranges_executor is None:
                import concurrent.futures

                self._cat_ranges_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._CAT_RANGES_MAX_WORKERS
                )
            return self._cat_ranges_executor

    def cat_ranges(
        self,
        paths: list[str],
        starts: list[int] | int,
        ends: list[int] | int,
        max_gap: int | None = None,
        on_error: str = "return",
        **kwargs: Any,
    ) -> list:
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
            # Marks this thread as occupying a pool slot, so a large read does
            # not try to pipeline into the pool it is running on.
            self._on_read_pool.active = True
            try:
                return self.cat_file(p, s, e)
            except Exception as exc:
                if on_error == "return":
                    return exc
                raise
            finally:
                self._on_read_pool.active = False

        workers = min(n, self._CAT_RANGES_MAX_WORKERS)
        if n <= 1 or workers <= 1:
            return [_read_one(p, s, e) for p, s, e in zip(paths, starts, ends)]

        results = list(self._executor.map(lambda args: _read_one(*args), zip(paths, starts, ends)))
        return results

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
            with self._lock.read():
                obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDWR)
                try:
                    B.dfs_punch(self._dfs, obj, size, B.DFS_MAX_FSIZE)
                finally:
                    B.dfs_release(obj)

        self._retry(_inner)
        self._last_activity = time.monotonic()
        self._invalidate_listing(path)

    # ── Checksum / du / streaming transfer ─────────────────────────────

    def checksum(self, path: str) -> str:
        """Return a checksum for the file at path.

        Uses the file's mode, size, and mtime as a lightweight change
        indicator. This is not a content hash: it changes when the file
        is recreated (truncate+write), but not on append (which DFS
        doesn't support anyway).

        Suitable for cache invalidation in HuggingFace Datasets and Dask.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()

        def _checksum_inner() -> str:
            with self._lock.read():
                obj, st = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                try:
                    return f"{st.mode}:{st.size}:{st.mtime:.6f}"
                finally:
                    B.dfs_release(obj)

        result = self._retry(_checksum_inner)
        self._last_activity = time.monotonic()
        return result

    def get_file(self, rpath: str, lpath: str, **kwargs):
        """Download a file from DFS to a local path.

        Streams in 16MB chunks to avoid buffering the entire file in memory.
        The lock is released between chunks so other operations can interleave.

        A directory creates ``lpath`` and returns, matching
        ``AbstractFileSystem.get_file``. That guard is what makes recursive
        ``get()`` work: fsspec walks the tree and calls this for every path,
        directories included. Without it a directory was read as a file and
        failed with EINVAL: and because the error quoted the directory's own
        size it looked like a short read, not a type error. That broke
        ``fs.get(recursive=True)`` and, through it, HuggingFace's
        ``load_from_disk()``.
        """
        self._check_closed()
        rpath = self._strip_protocol(rpath).rstrip("/")
        self._check_not_root(rpath or "/", "download")

        obj: Any
        with self._lock.read():
            generation = self._mount_generation
            obj, st = B.dfs_lookup_path(self._dfs, rpath, os.O_RDONLY)

        # Checked from the stat we already hold, rather than calling
        # self.isdir() like the base class does: that would cost an extra
        # round trip per entry on a recursive download.
        if st.is_dir:
            self._release_object(obj, generation)
            os.makedirs(lpath, exist_ok=True)
            self._last_activity = time.monotonic()
            return None

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=os.path.dirname(os.path.abspath(lpath)),
                prefix=".daos-download-",
                delete=False,
            ) as f:
                temporary = f.name
                offset = 0
                while offset < st.size:
                    with self._lock.read():
                        self._assert_same_mount(generation, "a download")
                        data = B.dfs_read(
                            self._dfs, obj, offset, min(B._CHUNK_SIZE, st.size - offset)
                        )
                    if not data:
                        raise OSError(errno.EIO, "premature EOF during download", rpath)
                    f.write(data)
                    offset += len(data)
            releasing, obj = obj, None
            self._release_object(releasing, generation)
            os.replace(temporary, lpath)
            temporary = None
        finally:
            try:
                self._release_object(obj, generation)
            finally:
                if temporary is not None:
                    os.unlink(temporary)
        self._last_activity = time.monotonic()

    def put_file(self, lpath: str, rpath: str, **kwargs):
        """Upload to a temporary sibling and replace the destination on success.

        The local source is opened before any remote mutation. A failed copy
        leaves the existing destination intact.
        """
        self._check_closed()
        if self._readonly:
            raise PermissionError("filesystem is read-only")
        rpath = self._strip_protocol(rpath).rstrip("/")
        self._check_not_root(rpath or "/", "upload")
        with open(lpath, "rb") as source:
            parent, name, generation = self._resolve_path(rpath)
            temporary = ".daos-upload-" + uuid.uuid4().hex
            obj = None
            created = False
            try:
                # Keep the mount alive until commit or cleanup. Other readers
                # can still run, but reconnect cannot strand a temporary file.
                with self._lock.read():
                    self._assert_same_mount(generation, "opening an upload")
                    try:
                        obj = B.dfs_open_file(
                            self._dfs,
                            parent,
                            temporary,
                            stat_mod.S_IFREG | 0o644,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            oclass_id=self._object_class,
                            chunk_size=self._chunk_size,
                        )
                        created = True
                        offset = 0
                        while True:
                            chunk = source.read(B._CHUNK_SIZE)
                            if not chunk:
                                break
                            self._assert_same_mount(generation, "an upload")
                            B.dfs_write(self._dfs, obj, offset, chunk)
                            offset += len(chunk)
                        self._assert_same_mount(generation, "committing an upload")
                        closing, obj = obj, None
                        B.dfs_close(closing)
                        B.dfs_move(self._dfs, parent, temporary, parent, name)
                        created = False
                    finally:
                        try:
                            if obj is not None:
                                self._release_object(obj, generation, close=True)
                        finally:
                            if created and generation == self._mount_generation:
                                B.dfs_remove(self._dfs, parent, temporary, force=False)
            finally:
                self._release_parent(parent, generation)
        self._last_activity = time.monotonic()
        self._invalidate_listing(rpath)

    # ── Teardown ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Tear down DFS mount and DAOS connections.

        Shutdown order matters:
        1. Set ``self._closed = True`` so other threads fail fast.
        2. Shut down the ``_cat_ranges_executor`` (wait for in-flight tasks),
           *outside* the lock: those workers take the shared lock themselves,
           so draining them while holding it would deadlock.
        3. Take the lock exclusively, which waits for remaining in-flight
           operations to drain.
        4. Release root handle (must happen before umount/disconnect).
        5. Disconnect from DAOS (umount/cont_close/pool_disconnect).
        6. Clear all handle attributes.
        """
        if self._forked or self._fork_generation != _fork.generation():
            # Inherited across a fork: the handles are already unusable and
            # calling into libdfs would hang. Drop them and let them leak
            # until this child exits; the parent owns the real teardown.
            self._closed = True
            self._root = self._dfs = self._coh = self._poh = None
            return

        # Phase 1: publish the close so no new operation starts.
        with self._lock.write():
            if self._closed:
                if self._cleanup_error:
                    raise RuntimeError(self._cleanup_error)
                return
            # Serialize with lazy executor creation. Publishing _closed while
            # holding this mutex means _executor can neither create a pool
            # after we detach the old one nor escape this shutdown cycle.
            with self._executor_lock:
                self._closed = True
                executor = self._cat_ranges_executor
                self._cat_ranges_executor = None

        _logger.debug(
            "Closing DAOS connection pool=%s container=%s",
            self._pool_label,
            self._cont_label,
        )

        # Phase 2: drain cat_ranges workers with no lock held.
        if executor is not None:
            executor.shutdown(wait=True)

        # Phase 3: exclusive teardown; waits for in-flight readers to finish.
        with self._lock.write():
            self._teardown_handles()

    def _teardown_handles(self) -> None:
        """Release the root handle, then umount and disconnect.

        **The exclusive lock must already be held.** Umounting while another
        thread is mid-call on this mount is undefined, which is what the
        exclusive acquire in the callers is for.

        Factored out of close() so the finaliser path shares one definition of
        the teardown order rather than keeping a second copy that can drift.
        """
        if self._cleanup_error is not None or B._runtime.failure:
            self._cleanup_error = self._cleanup_error or B._runtime.failure
            raise RuntimeError(self._cleanup_error)
        try:
            for reference in list(self._open_files.values()):
                file = reference()
                if file is None:
                    raise RuntimeError("open file collected without successful cleanup")
                file._discard_for_teardown()
            if self._root is not None:
                B.dfs_release(self._root)
                self._root = None
            if self._uses_dfs_connect and B._HAS_DFS_CONNECT and self._dfs is not None:
                _, libdfs = B._ensure_libs()
                B._check(libdfs.dfs_disconnect(self._dfs), "dfs_disconnect")
                self._dfs = None
            else:
                steps: list[tuple[Callable[[Any], None], str]] = [
                    (B.dfs_umount, "_dfs"),
                    (B.cont_close, "_coh"),
                    (B.pool_disconnect, "_poh"),
                ]
                for function, attr in steps:
                    handle = getattr(self, attr)
                    if handle is not None:
                        function(handle)
                        setattr(self, attr, None)
        except BaseException as exc:
            self._cleanup_error = f"filesystem cleanup failed; refusing destructive retry: {exc}"
            raise
        B._runtime.disconnected(self)

    def _close_from_finalizer(self) -> bool:
        """Best-effort teardown from ``__del__``. Never blocks, never raises.

        Returns True if the handles were actually released.

        ``__del__`` used to do nothing but set ``_closed``, on the grounds that
        "DAOS C calls can segfault" during GC. That is too strong a conclusion:
        DFSFile.__del__ has always called dfs_close/dfs_release from the same
        position and is fine, because it is *guarded*. What is genuinely unsafe
        is not the C call; it is blocking, and running when the mount is in
        use or already torn down. So this guards those three things instead of
        giving up:

        * **Interpreter shutdown.** ``B`` may be partly collected by now, so
          the attribute check comes first. (``pool.py`` imports the bindings
          function-locally for the same reason.)
        * **Fork.** Handles inherited across a fork are unusable and calling
          into libdfs would hang. The parent owns the real teardown.
        * **Blocking.** The exclusive lock is tried with ``timeout=0``, never
          waited on. A finaliser runs on whichever thread happened to trigger
          collection, so blocking here can stall an unrelated thread, or hang
          during interpreter shutdown when the holder will never run again.
          If the lock is unavailable the handles leak until process exit --
          exactly what happened on *every* path before, so this is strictly
          better and never worse.

        The cost of doing nothing was a real leak: a filesystem dropped without
        close() held its pool connection, container handle and mount open for
        the life of the process. Bounded for get_fs() users, unbounded for code
        that builds a filesystem per task.
        """
        # Interpreter shutdown: the bindings may already be torn down.
        if B is None or not hasattr(B, "dfs_release"):
            return False
        # Only tear down through the library that created these handles.
        if getattr(self, "_bindings", None) is not B:
            return False
        if getattr(self, "_closed", True):
            return False
        if self._forked or self._fork_generation != _fork.generation():
            # Fork: drop the references, leave the handles to the parent.
            self._closed = True
            self._root = self._dfs = self._coh = self._poh = None
            return False

        try:
            acquired = self._lock.acquire_write(timeout=0)
        except RuntimeError:
            # Collection fired on a thread already holding the shared lock.
            # acquire_write() refuses the upgrade rather than deadlocking;
            # leaking is the only safe answer from inside someone else's call.
            return False
        except Exception:
            return False
        if not acquired:
            return False

        try:
            self._closed = True
            # The cat_ranges workers take the shared lock themselves, so an
            # in-flight task would have blocked the acquire above. Holding the
            # lock means they are idle; shut down without waiting, because
            # waiting is the one thing a finaliser must not do.
            executor = getattr(self, "_cat_ranges_executor", None)
            self._cat_ranges_executor = None
            self._teardown_handles()
        finally:
            self._lock.release_write()

        if executor is not None:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        return True

    def __del__(self) -> None:
        """Safety net: release handles if close() was never called."""
        try:
            self._close_from_finalizer()
        except Exception:
            pass
        try:
            self._closed = True
        except Exception:
            pass
