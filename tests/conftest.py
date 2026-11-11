# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for conftest."""

import contextlib
import gc
import importlib
import os
import uuid
from unittest.mock import patch as mock_patch

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires a running DAOS cluster")


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
    Close explicitly after the session has finished its work.
    """
    from daos_fsspec.core import DFSFileSystem

    fs = DFSFileSystem(pool=daos_pool, container=daos_container)
    yield fs
    fs.close()


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


# ── Binding patch helper ─────────────────────────────────────────────────────
#
# Every daos_fsspec module does `from daos_fsspec import _bindings as B`, which
# binds a module-level name. Patching one module therefore does NOT affect the
# others; and that is not theoretical: splitting the listing code out of
# core.py into _listing.py silently broke 34 tests, because the moved methods
# kept using the real bindings and tried to dlopen libdaos.
#
# Patch from one place so the next module split costs nothing. The guard test in
# tests/test_binding_patch_targets.py fails if a module imports B without being
# registered here, so a silent gap becomes a failing test.

BINDING_MODULES = (
    "daos_fsspec.core",
    "daos_fsspec._listing",
    "daos_fsspec.dfs_file",
)
# Note on pool.py: it imports the bindings *inside a function* (deliberately --
# the comment there explains that _bindings can be garbage-collected during
# interpreter shutdown). A function-local import creates no module attribute, so
# there is nothing for patch() to replace; code like that has to be mocked by
# patching daos_fsspec._bindings itself. The guard test only considers
# module-level importers for that reason.


@contextlib.contextmanager
def patch_bindings(mock_B):
    """Patch the ctypes binding layer across every module that imports it.

    Collects garbage before restoring the real bindings. A mock-built
    DFSFileSystem or DFSFileSystemPool that is left to be finalised later runs
    its cleanup; dfs_release, dfs_umount, cont_close, pool_disconnect --
    against whatever ``B`` is at that moment. Offline that is the real module
    with no libdaos to find, so it prints a traceback and nothing worse, which
    is why this used to look like harmless noise.

    On a machine with a cluster it is not harmless. DFSFileSystemPool.__del__
    fired during a GC pause in the middle of an unrelated real
    ``dfs_readdirplus``, called ``B.dfs_release`` while ``B`` was still a mock,
    and allocating mock children with a libdfs call in flight segfaulted the
    whole live suite several tests downstream from the one that leaked.

    Collecting here makes those finalisers run while the mock is still
    installed, so the worst case is a few harmless mock calls. Tests should
    still close what they construct; this is a backstop, not a licence.
    """
    # Mock call histories retain arguments strongly; a mocked lifecycle
    # registration would keep filesystems alive and invalidate GC tests.
    from types import SimpleNamespace

    mock_B._runtime = SimpleNamespace(
        connection=lambda owner: contextlib.nullcontext(),
        disconnected=lambda owner: None,
        failure=None,
    )
    with contextlib.ExitStack() as stack:
        for module_name in BINDING_MODULES:
            # mock.patch("pkg.submodule.attr") walks attributes from ``pkg``;
            # it does not reliably import a submodule that has not yet been
            # exposed on the package. Import explicitly so a single test module
            # is runnable without depending on collection order.
            module = importlib.import_module(module_name)
            stack.enter_context(mock_patch.object(module, "B", mock_B))
        try:
            yield mock_B
        finally:
            gc.collect()


@pytest.fixture
def patched_bindings():
    """Fixture form of :func:`patch_bindings`, for tests that want a factory."""
    return patch_bindings
