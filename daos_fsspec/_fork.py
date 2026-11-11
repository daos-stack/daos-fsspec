# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Reject inherited DAOS state after fork. Native threads and locks cannot be reused in a child; use the spawn multiprocessing start method."""

from __future__ import annotations

import os
import threading
import weakref
from typing import Any

# Bumped every time this process is forked. Instances capture the value at
# construction; a mismatch means "you are in a child of the process that
# created me". An int compare keeps this off the hot path's cost profile.
_generation = 0
_fork_parent_pid: int | None = None

# Live objects holding DAOS handles, so the child can poison them all at once.
#
# Keyed by id() rather than using a WeakSet: fsspec's AbstractFileSystem
# hashes and compares by _fs_token, so two independent connections to the same
# pool/container are "equal" and would collide in a set: we need every live
# instance, by identity. (DFSFileSystemPool._in_use does this for the same
# reason.) The value is a weakref so registration never keeps handles alive.
_live: dict[int, weakref.ref[Any]] = {}
_registry_lock = threading.Lock()


def generation() -> int:
    """Current fork generation for this process."""
    return _generation


def _check_can_initialize() -> None:
    """Reject creation of new DAOS state in a process produced by ``fork``.

    Poisoning inherited filesystem objects is not enough. libdaos itself has
    process-global state and background threads; constructing a fresh Python
    object in the child would reuse that inherited runtime (``daos_init`` is
    already marked initialized) and can hang on its orphaned locks.
    """
    if _generation == 0:
        return
    parent = _fork_parent_pid if _fork_parent_pid is not None else "its parent"
    raise RuntimeError(
        f"DAOS cannot be initialized in forked child {os.getpid()} (forked from "
        f"{parent}). The inherited DAOS runtime cannot be repaired, even by "
        "constructing a new DFSFileSystem or calling get_fs(). Use the 'spawn' "
        "start method and construct the filesystem in the spawned process."
    )


def register(obj: Any) -> None:
    """Track an object holding DAOS handles.

    ``obj`` must implement ``_poison_after_fork()``, called in the child.
    """
    key = id(obj)

    def _drop(ref: weakref.ref[Any], key: int = key) -> None:
        # Runs during deallocation, at arbitrary points including GC. Compare
        # identity before removing so a recycled id() cannot evict its
        # successor. dict ops are atomic, so this needs no lock: and must
        # not take one, since a finalizer can fire while it is held.
        if _live.get(key) is ref:
            _live.pop(key, None)

    with _registry_lock:
        _live[key] = weakref.ref(obj, _drop)


def fork_error(created_pid: int) -> RuntimeError:
    """Build the error raised when DAOS handles are touched after a fork."""
    return RuntimeError(
        f"DAOS handles do not survive fork(): this object was created in "
        f"process {created_pid} and is being used in forked child "
        f"{os.getpid()}. The DAOS client runs background threads that fork() "
        f"does not duplicate, so any call made here would hang indefinitely "
        f"rather than fail.\n"
        f"Use the 'spawn' start method instead, and create the filesystem "
        f"inside the child:\n"
        f"    multiprocessing.set_start_method('spawn')\n"
        f"    torch.utils.data.DataLoader(..., multiprocessing_context='spawn')\n"
        f"See daos_fsspec.get_fs() for a per-process filesystem cache that is "
        f"safe to call from spawned workers."
    )


def _after_in_child() -> None:
    """Poison every inherited DAOS handle. Runs in the child after fork()."""
    global _fork_parent_pid, _generation, _registry_lock

    _generation += 1
    _fork_parent_pid = os.getppid()

    # The registry lock may have been held by a thread that does not exist in
    # this child. Replace it rather than risk deadlocking on it. Safe: the
    # child is single-threaded at this point.
    _registry_lock = threading.Lock()

    try:
        refs = list(_live.values())
    except Exception:  # pragma: no cover - registry mid-update at fork time
        return

    for ref in refs:
        obj = ref()
        if obj is None:
            continue
        try:
            obj._poison_after_fork()
        except Exception:  # pragma: no cover - best effort; never raise in a fork hook
            pass


# Unix only; DAOS has no Windows client, so absence just means nothing to do.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_in_child)
