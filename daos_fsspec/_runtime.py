# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Cold-path ownership of DAOS/DFS initialization and process exit cleanup.

Applications must stop their work before closing resources or finalizing the
runtime. This module does not admit, count, or drain individual I/O operations.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from daos_fsspec import _fork


class Runtime:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.libraries: tuple[Any, Any] | None = None
        self.daos = False
        self.dfs = False
        self.stopping = False
        self.finished = False
        self.failure: str | None = None
        self.registered = False
        self.connecting = 0
        self.initializing = False
        # No weakref removal callback: an owner collected without successful
        # cleanup must remain visible, so fini cannot overtake leaked handles.
        self.mounts: dict[int, weakref.ReferenceType[Any]] = {}

    def _check(self) -> None:
        if sys.is_finalizing():
            raise RuntimeError("DAOS initialization is unavailable during interpreter finalization")
        _fork._check_can_initialize()
        if os.getpid() != self.pid:
            raise _fork.fork_error(self.pid)
        if self.failure or self.stopping or self.finished:
            raise RuntimeError(self.failure or "DAOS runtime has been finalized or is stopping")

    def initialize(self, *, dfs: bool = False) -> None:
        self._check()  # before a lock potentially inherited across fork
        with self.lock:
            self._check()
            if self.initializing:
                raise RuntimeError("recursive DAOS initialization")
            self.initializing = True
            try:
                from daos_fsspec import _bindings as bindings

                if self.libraries is None:
                    self.libraries = bindings._ensure_libs()
                daos, libdfs = self.libraries
                for name, library, owned in [
                    ("daos_init", daos, "daos"),
                    ("dfs_init", libdfs, "dfs"),
                ]:
                    if (owned == "dfs" and not dfs) or getattr(self, owned):
                        continue
                    try:
                        rc = getattr(library, name)()
                    except BaseException as exc:
                        self.failure = f"{name} interrupted; native ownership is uncertain: {exc}"
                        raise
                    bindings._check(rc, name)
                    setattr(self, owned, True)
                    if not self.registered:
                        try:
                            atexit.register(self.at_exit)
                        except BaseException as exc:
                            self.failure = f"cannot register DAOS exit cleanup: {exc}"
                            raise
                        self.registered = True
            finally:
                self.initializing = False

    @contextmanager
    def connection(self, owner: Any) -> Iterator[None]:
        """Publish connection ownership without serializing independent connects."""
        self._check()
        with self.lock:
            self._check()
            previous = self.mounts.get(id(owner))
            if previous is not None and previous() is not owner:
                raise RuntimeError("an earlier filesystem was collected without cleanup")
            owner._runtime_reference = weakref.ref(owner)
            self.mounts[id(owner)] = owner._runtime_reference
            self.connecting += 1
        try:
            yield
        except BaseException:
            if not owner._cleanup_error and all(
                getattr(owner, name, None) is None for name in ("_dfs", "_coh", "_poh")
            ):
                self.disconnected(owner)
            raise
        finally:
            with self.lock:
                self.connecting -= 1

    def disconnected(self, owner: Any) -> None:
        # Called only after successful native teardown. A finalizer must not
        # wait for another thread holding the runtime lock; dict removal needs
        # no lock on the supported GIL-enabled CPython interpreters.
        reference = self.mounts.get(id(owner))
        # Weakrefs may already be cleared during cyclic GC. Compare the
        # registration token, so a recycled id cannot erase an orphan record.
        if reference is not None and reference is getattr(owner, "_runtime_reference", None):
            self.mounts.pop(id(owner), None)

    def finish(self, *, automatic: bool = False) -> None:
        if os.getpid() != self.pid or _fork.generation():
            return  # never touch inherited native state or locks
        with self.lock:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.finished:
                return
            if self.stopping:
                raise RuntimeError("DAOS runtime finalization is already in progress")
            if self.initializing or self.connecting:
                raise RuntimeError("stop connection creation before finalizing DAOS")
            if self.mounts and not automatic:
                raise RuntimeError(
                    "stop application work and close all filesystems before daos_fini"
                )
            self.stopping = True
            owners = [ref() for ref in list(self.mounts.values())]
        # Normal process exit occurs after application/non-daemon workers have
        # finished. Do not hold the runtime lock while close joins an executor.
        try:
            for owner in owners:
                if owner is None:
                    raise RuntimeError(
                        "filesystem collected without successful cleanup; refusing fini"
                    )
                owner.close()
            if self.mounts:
                raise RuntimeError("filesystem cleanup incomplete; refusing native finalization")
            if self.libraries is not None:
                from daos_fsspec import _bindings as bindings

                daos, libdfs = self.libraries
                for name, library, owned in [
                    ("dfs_fini", libdfs, "dfs"),
                    ("daos_fini", daos, "daos"),
                ]:
                    if getattr(self, owned):
                        bindings._check(getattr(library, name)(), name)
                        setattr(self, owned, False)
            self.finished = True
        except BaseException as exc:
            # Even a nonzero fini result can accompany partial native cleanup.
            # Retain the failure and never retry destructive teardown.
            self.failure = f"DAOS finalization failed: {exc}"
            raise
        finally:
            self.stopping = False

    def at_exit(self) -> None:
        try:
            self.finish(automatic=True)
        except Exception:
            logging.getLogger(__name__).error("DAOS exit cleanup failed", exc_info=True)
