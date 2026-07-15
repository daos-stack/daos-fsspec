# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for daos-fsspec.

Tests marked with @pytest.mark.live require a running DAOS cluster.
Set DAOS_FSSPEC_TEST_POOL and DAOS_FSSPEC_TEST_CONTAINER environment variables.

The live fixtures create a test directory, yield the filesystem, and clean up.
"""

import contextlib
import os
import uuid

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires a running DAOS cluster")


class _HandleTracker:
    """Counts DFS object-handle acquisitions vs. releases through the binding
    layer, so a leak shows up as a non-zero net at the end of a workload.

    Wraps the ``_bindings`` functions that return a handle (+1) and those that
    release one (-1). Every object-handle release funnels through
    ``dfs_release``: callers use it directly, and ``dfs_close`` is the alias
    ``def dfs_close(obj): dfs_release(obj)``. That inner ``dfs_release`` name
    resolves through the ``_bindings`` module globals — which is exactly what
    ``setattr(B, "dfs_release", ...)`` rebinds — so wrapping ``dfs_release``
    alone already counts closes. Wrapping ``dfs_close`` too would double-count
    every close (once for the wrapper, once for its inner ``dfs_release``).
    """

    _ACQUIRE = ("dfs_lookup_path", "dfs_lookup_rel_path", "dfs_open_file", "dfs_create_symlink")
    _RELEASE = ("dfs_release",)

    def __init__(self):
        self.acquired = 0
        self.released = 0

    @property
    def net(self):
        """Handles acquired but not yet released. 0 == balanced (no leak)."""
        return self.acquired - self.released

    def _wrap_acquire(self, orig):
        def w(*a, **k):
            result = orig(*a, **k)  # only count on success — a raise acquires nothing
            self.acquired += 1
            return result

        return w

    def _wrap_release(self, orig):
        def w(*a, **k):
            self.released += 1
            return orig(*a, **k)

        return w

    @contextlib.contextmanager
    def track(self):
        from daos_fsspec import _bindings as B

        saved = {}
        try:
            for name in self._ACQUIRE:
                saved[name] = getattr(B, name)
                setattr(B, name, self._wrap_acquire(saved[name]))
            for name in self._RELEASE:
                saved[name] = getattr(B, name)
                setattr(B, name, self._wrap_release(saved[name]))
            yield self
        finally:
            for name, orig in saved.items():
                setattr(B, name, orig)


@pytest.fixture
def handle_tracker():
    """Fresh handle-leak tracker; use ``with handle_tracker.track(): ...``."""
    return _HandleTracker()


def pytest_collection_modifyitems(config, items):
    pool = os.environ.get("DAOS_FSSPEC_TEST_POOL")
    container = os.environ.get("DAOS_FSSPEC_TEST_CONTAINER")
    if not pool or not container:
        skip = pytest.mark.skip(reason="DAOS_FSSPEC_TEST_POOL/CONTAINER not set")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def daos_pool():
    return os.environ.get("DAOS_FSSPEC_TEST_POOL", "")


@pytest.fixture(scope="session")
def daos_container():
    return os.environ.get("DAOS_FSSPEC_TEST_CONTAINER", "")


@pytest.fixture(scope="session")
def _dfs_connection(daos_pool, daos_container):
    """Shared DFSFileSystem connection for the entire test session.

    Session-scoped to avoid segfaults from repeated pool_connect/pool_disconnect
    cycles and GC finalization of DAOS handles across test boundaries.
    Not closed on teardown — handles leak until process exit, which is safe.
    """
    from daos_fsspec.core import DFSFileSystem

    fs = DFSFileSystem(pool=daos_pool, container=daos_container)
    yield fs
    # Intentionally NOT calling fs.close() — the process will clean up.
    # Closing a session-scoped handle while ThreadPoolExecutor daemon threads
    # are still alive causes segfaults during GC/teardown.


@pytest.fixture
def dfs_fs(_dfs_connection):
    """Yield a DFSFileSystem with a unique test directory per test."""
    fs = _dfs_connection
    test_dir = f"/fsspec-test-{uuid.uuid4().hex[:8]}"
    fs.mkdir(test_dir)
    fs._test_dir = test_dir
    yield fs
    # Cleanup
    try:
        fs.rm(test_dir, recursive=True)
    except Exception:
        pass
