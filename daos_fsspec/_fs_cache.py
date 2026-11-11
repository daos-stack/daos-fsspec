# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Process-local filesystem cache. Connections are keyed by process and configuration; forked children must use spawn before initializing DAOS."""

from __future__ import annotations

import os
import threading
from typing import Any

from daos_fsspec import _fork
from daos_fsspec.core import DFSFileSystem

__all__ = ["cache_info", "clear_fs_cache", "get_fs"]

# (pid, fork_generation, pool, container, sorted kwargs) -> DFSFileSystem
_cache: dict[tuple[Any, ...], DFSFileSystem] = {}

# Held across construction, not just the dict lookup. That serialises
# connection setup, which is the point: two threads racing on the same key must
# produce one connection, not two with one leaked. Different keys serialise too,
# which is a deliberate trade; connecting is rare, and silently doubling
# handle count is the failure mode worth preventing.
_lock = threading.Lock()


class _ForkGuard:
    """Resets this module's state in a forked child.

    ``_lock`` is held across connection setup; deliberately, so two threads
    racing on one key produce a single connection. That also makes it the
    widest possible window for a fork to land inside, and a child inheriting it
    locked blocks forever on the first ``get_fs()``: the holder is a thread
    that does not exist in the child. ``_fork`` already replaces its own
    registry lock and ``DFSFileSystem``/``DFSFileSystemPool`` replace theirs;
    this was the one left out, which is precisely the silent hang ``_fork``
    exists to eliminate; in the very API the fork error message recommends.

    Registered through the normal ``_fork`` registry rather than a new
    mechanism; the module keeps the instance alive because the registry holds
    only weak references.
    """

    def _poison_after_fork(self) -> None:
        global _lock
        _lock = threading.Lock()
        # Every inherited entry is keyed to the parent's pid and fork
        # generation, so none can ever be returned. Dropping them releases the
        # poisoned filesystems; their finalisers see the generation change and
        # skip libdfs rather than releasing handles the parent still owns.
        _cache.clear()


_fork_guard = _ForkGuard()
_fork.register(_fork_guard)


def _key(pool: str, container: str, kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Build the cache key, rejecting kwargs that cannot form one."""
    try:
        extra = tuple(sorted(kwargs.items()))
        key = (os.getpid(), _fork.generation(), pool, container, extra)
        hash(key)
    except TypeError as exc:
        raise TypeError(
            "get_fs() kwargs must be hashable so connections can be cached by "
            f"their arguments; got {kwargs!r}. Construct DFSFileSystem directly "
            "if you need unhashable options."
        ) from exc
    return key


def _prune_foreign_entries(current_pid: int) -> None:
    """Drop entries belonging to other processes.

    After ``fork()`` the child inherits a copy of the whole dict. Those entries
    can never be returned; their pid no longer matches; but they keep the
    poisoned filesystems alive. Dropping the references is both a leak fix and
    correct ownership: the parent still owns the real handles and is the only
    side that may tear them down.
    """
    stale = [k for k in _cache if k[0] != current_pid]
    for k in stale:
        del _cache[k]


def get_fs(pool: str, container: str, **kwargs: Any) -> DFSFileSystem:
    """Return a process-local cached :class:`DFSFileSystem`.

    Repeated calls with the same arguments in the same process return the same
    instance. A process created with ``spawn`` gets its own. A forked child is
    rejected because the inherited DAOS runtime cannot be repaired.

    This is the supported pattern for multiprocessing ``DataLoader`` workers
    under ``spawn``: keep only the ``pool``/``container`` strings on the dataset
    (strings pickle; DAOS handles do not) and call ``get_fs`` inside
    ``__getitem__`` or ``worker_init_fn``.

    Parameters
    ----------
    pool, container
        DAOS pool and container labels or UUIDs.
    **kwargs
        Passed to :class:`DFSFileSystem`. Must be hashable, as they form part
        of the cache key.

    Returns
    -------
    DFSFileSystem
        A connected filesystem, cached for this process.

    Examples
    --------
    ::

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, pool, container, files):
                # strings only; no handle on self, so this pickles
                self.pool, self.container, self.files = pool, container, files

            def __getitem__(self, i):
                fs = daos_fsspec.get_fs(self.pool, self.container)
                return fs.cat_file(self.files[i])
    """
    # Reject before taking the cache lock. The at-fork hook replaces that lock,
    # but libdaos's inherited locks and initialized flag remain poisoned.
    _fork._check_can_initialize()
    pid = os.getpid()
    key = _key(pool, container, kwargs)

    with _lock:
        _prune_foreign_entries(pid)

        fs = _cache.get(key)
        # is_alive covers both closed and fork-poisoned. A cached-but-closed
        # filesystem must be replaced rather than handed back, otherwise the
        # first caller to close() breaks every later get_fs() in the process.
        if fs is not None and fs.is_alive:
            return fs

        if fs is not None:
            del _cache[key]

        fs = DFSFileSystem(pool=pool, container=container, **kwargs)
        _cache[key] = fs
        return fs


def clear_fs_cache(*, close: bool = False) -> int:
    """Drop every cached filesystem for this process. Returns the count removed.

    Parameters
    ----------
    close
        If True, call ``close()`` on each before dropping it. Off by default:
        another part of the program may still hold a reference to a cached
        filesystem, and closing it underneath them would be surprising. Pass
        True in tests, or at shutdown when nothing else is using them.
    """
    with _lock:
        entries = list(_cache.values())
        count = len(_cache)
        _cache.clear()

    if close:
        for fs in entries:
            try:
                fs.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
    return count


def cache_info() -> dict[str, Any]:
    """Introspection for debugging the 'why do I have N connections' question."""
    with _lock:
        return {
            "pid": os.getpid(),
            "fork_generation": _fork.generation(),
            "entries": len(_cache),
            "keys": [
                {"pid": k[0], "fork_generation": k[1], "pool": k[2], "container": k[3]}
                for k in _cache
            ],
        }
