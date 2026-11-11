# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Directory listing and traversal for :class:`~daos_fsspec.core.DFSFileSystem`.

Split out of ``core.py``, where these methods were spread across five separate
sections; "Listings cache", "Directory listing", "Walk", "Optimized bulk
reads" and "Checksum / du". That scattering had a real cost: it is why nothing
in the file made it apparent that ``find()``, ``walk()``, ``find_parallel()``
and ``du()`` each issue ``readdirplus`` directly and therefore bypass ``ls()``
entirely. A listings cache was added to ``ls()`` on the assumption that the
others composed on top of it. They do not.

So the grouping is the point. Everything that reads a directory now lives in one
file, and the next reader can see all of it at once.

This is a mixin rather than a separate collaborator because ``DFSFileSystem``
must stay a single class: fsspec registers it by protocol, and the
shared/exclusive lock is per-instance state these methods share with the rest of
the filesystem. Nothing here holds state of its own; every method reaches
through ``self`` for ``_dfs``, ``_lock``, ``_retry``, ``dircache`` and friends,
exactly as it did when it lived in ``core.py``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from daos_fsspec import _bindings as B

if TYPE_CHECKING:  # pragma: no cover - typing only
    from daos_fsspec._rwlock import RWLock

_logger = logging.getLogger(__name__)


def _entry_info(name: str, entry: B.DirEntry, type_hint: str) -> dict[str, Any]:
    """Build an fsspec info dict from one directory entry.

    Was defined twice, identically, inside ``find()`` and ``find_parallel()``.

    Note the missing-stat branch. It fabricates ``size: 0`` rather than failing,
    which is the same pattern ``ls()`` carries, and profiling
    (``docker/profile_ls.py``) never reached it on 2.6.5; ``dfs_readdirplus``
    stats every entry, and the ENOENT fallback skips entries it cannot stat
    rather than emitting ``stat=None``. Keeping all of that fabrication in one
    function means that if the assumption ever stops holding, there is a single
    place to make it loud.
    """
    st = entry.stat
    if st is None:
        return {"name": name, "size": 0, "type": type_hint}
    return {
        "name": name,
        # The stat's own value, for every type. This used to be
        # `st.size if type_hint == "file" else 0`, which made find(detail=True)
        # the only one of ls()/info()/find() that discarded a directory's size:
        # the same stat through two paths gave two answers (4096 vs 0). The
        # docstring above justifies fabricating a size when the stat is
        # *missing*, which is a different situation and still handled above.
        "size": st.size,
        "type": type_hint,
        "mode": st.mode,
        "mtime": st.mtime,
        "atime": st.atime,
        "uid": st.uid,
        "gid": st.gid,
    }


class DFSListingMixin:
    """Directory listing, traversal and the listings cache.

    Mixed into :class:`~daos_fsspec.core.DFSFileSystem`, which supplies the
    attributes declared below. Not useful standalone.
    """

    # --- contract required from the host class -------------------------------
    # Annotations only: no assignment, so nothing exists at runtime and
    # DFSFileSystem's real attributes are used unchanged. Declared so the
    # coupling is checkable rather than implicit, and so mypy can verify these
    # methods against it.
    _dfs: Any
    _root: Any
    _mount_generation: int
    _lock: RWLock
    dircache: Any
    _last_activity: float

    # Methods declared as callable attributes rather than stub defs: a stub
    # body of `...` with a non-None return type trips mypy's empty-body check,
    # and making them abstract would need ABCMeta, which conflicts with
    # fsspec's own metaclass on AbstractFileSystem.
    _check_closed: Callable[[], None]
    _assert_same_mount: Callable[[int, str], None]
    _retry: Callable[..., Any]
    info: Callable[..., dict]
    _strip_protocol: Callable[..., str]

    # ── Listings cache ───────────────────────────────────────────────────
    #
    # Consulted and populated by ls() only. find(), walk(), find_parallel() and
    # du() deliberately do NOT read it: each issues one readdirplus per
    # directory and skips the walk->ls->info chain entirely, so routing them
    # through the cache would undo that optimisation and, worse, let a
    # recursive traversal mix fresh and up-to-a-second-stale directories in one
    # result. They stay always-fresh. The practical consequence is that
    # ls() is cached and the recursive walks are not, which is the right way
    # round: ls() on one directory is what gets called repeatedly.

    # --- one place that opens a directory and reads it -----------------------

    def _readdirplus_dir(self, path: str) -> list[B.DirEntry]:
        """Read one directory's entries, with stats, in a single batch.

        The root is special and the distinction matters: ``self._root`` is a
        long-lived handle owned by the filesystem and must **not** be released,
        while every other path gets a per-call handle that must be. That
        branch was written out by hand at five call sites; five chances to
        either leak a handle or release the shared root, and releasing the root
        would corrupt every later operation on the mount.

        The lock is held across lookup, read and release, exactly as each of
        those sites did. Callers that need retry wrap this in ``self._retry``.
        """
        with self._lock.read():
            if path == "/":
                obj = self._root
                need_release = False
            else:
                obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
                need_release = True
            try:
                return B.dfs_readdirplus(self._dfs, obj)
            finally:
                if need_release:
                    B.dfs_release(obj)

    def _readdirplus_on(self, obj: Any, generation: int) -> list[B.DirEntry]:
        """Read entries from an already-open directory handle.

        For callers that opened the handle themselves and still need it
        afterwards; the parallel path falls back this way rather than looking
        the same directory up twice.
        """
        with self._lock.read():
            self._assert_same_mount(generation, "reading a directory")
            return B.dfs_readdirplus(self._dfs, obj)

    def _open_dir_handle(self, path: str) -> tuple[Any, bool, int]:
        """Open a directory and report whether the caller must release it."""
        with self._lock.read():
            generation = self._mount_generation
            if path == "/":
                return self._root, False, generation
            obj, _ = B.dfs_lookup_path(self._dfs, path, os.O_RDONLY)
            return obj, True, generation

    def _listings_cache_get(self, path: str) -> list[dict] | None:
        """Return a cached detailed listing, or None on miss/disabled."""
        try:
            cached: Any = self.dircache.get(path)
        except Exception:  # noqa: BLE001 - cache must never break a listing
            return None
        return cached if cached is not None else None

    def _listings_cache_put(self, path: str, detailed: list[dict]) -> None:
        """Cache a detailed listing. Never raises."""
        try:
            self.dircache[path] = detailed
        except Exception:  # noqa: BLE001 - cache must never break a listing
            _logger.debug("listings cache store failed for %s", path, exc_info=True)

    def _invalidate_listing(self, *paths: str) -> None:
        """Drop cached listings for the parent directory of each path.

        Called after every local mutation. Invalidating the *parent* is the
        point: creating ``/a/b/c.bin`` changes the listing of ``/a/b``, not of
        ``/a/b/c.bin``. Both are dropped anyway, since a path that was a
        directory may itself have been removed or replaced.
        """
        if not paths:
            return
        try:
            for path in paths:
                if not path:
                    continue
                p = self._strip_protocol(path).rstrip("/") or "/"
                self.dircache.pop(p, None)
                parent = p.rsplit("/", 1)[0] or "/"
                self.dircache.pop(parent, None)
        except Exception:  # noqa: BLE001 - invalidation must never raise
            # A cache that cannot be invalidated must not be trusted.
            try:
                self.dircache.clear()
            except Exception:
                pass

    def invalidate_cache(self, path: str | None = None) -> None:
        """Drop cached listings. With no argument, drops everything.

        This is the fsspec-documented escape hatch, and the supported way to
        pick up a peer rank's writes without waiting out the TTL.
        """
        if path is None:
            self.dircache.clear()
        else:
            self._invalidate_listing(path)

    # ── Directory listing ────────────────────────────────────────────────

    def ls(self, path: str, detail: bool = False, **kwargs: Any) -> list:
        """List a directory.

        Listings are **not** cached unless ``use_listings_cache=True`` was
        passed to the constructor; see the class docstring for why that is the
        default and what the cache does and does not guarantee.
        """
        self._check_closed()
        path = self._strip_protocol(path).rstrip("/") or "/"

        # Always cache the detailed form and derive the name-only form from it,
        # which is what fsspec's own backends do: caching the two shapes
        # separately would double the round trips and let them disagree.
        cached = self._listings_cache_get(path)
        if cached is not None:
            return cached if detail else [e["name"] for e in cached]

        entries = self._retry(self._readdirplus_dir, path)
        self._last_activity = time.monotonic()

        detailed: list[dict] = []
        complete = True
        for entry in entries:
            full_path = path.rstrip("/") + "/" + entry.name
            if full_path.startswith("//"):
                full_path = full_path[1:]
            if entry.stat:
                detailed.append(
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
                # No stat for this entry. Profiling on 2.6.5 never reached this
                # branch (docker/profile_ls.py); dfs_readdirplus parses a stat
                # for every entry, and the ENOENT fallback skips entries it
                # cannot stat rather than emitting stat=None. But the values
                # below are fabricated, not measured, so the listing must not be
                # cached: a one-second lie is worse than a round trip, and
                # caching it would make an invented size:0 look authoritative.
                complete = False
                detailed.append(
                    {
                        "name": full_path,
                        "size": 0,
                        "type": "file",
                    }
                )

        if complete:
            self._listings_cache_put(path, detailed)

        return detailed if detail else [e["name"] for e in detailed]

    # ── Walk (optimized: uses readdirplus, avoids extra stat) ───────────

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
        entries = self._retry(self._readdirplus_dir, path)
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

    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Recursively list all files under *path*.

        Uses a single readdirplus call per directory (with batch stat),
        avoiding the walk→ls→info chain the base class uses.
        """
        path = self._strip_protocol(path).rstrip("/") or "/"
        self._check_closed()
        out = {}

        def _readdir(dir_path, depth):
            entries = self._retry(self._readdirplus_dir, dir_path)
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

    def find_parallel(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> list[str] | dict[str, Any]:
        """Compatibility alias for find(). num_workers is ignored; traversal is serial."""
        if num_workers != 4:
            _logger.debug(
                "find_parallel(num_workers=%d) ignored; this is now an alias for find()",
                num_workers,
            )
        result: list[str] | dict[str, Any] = self.find(
            path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs
        )
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
            entries = self._retry(self._readdirplus_dir, dir_path)
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
