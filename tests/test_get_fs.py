# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for get fs."""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import patch_bindings

from daos_fsspec import _fork, _fs_cache


class _FakeDaosError(Exception):
    is_retryable = False


@pytest.fixture
def mock_bindings():
    """Patch the ctypes layer so constructing a DFSFileSystem is cheap.

    Yields the mock so tests can count how many real constructions happened -
    which is the assertion that stops these tests being vacuous.
    """
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.pool_connect.return_value = MagicMock(name="poh")
    mock_B.cont_open.return_value = MagicMock(name="coh")
    mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755),
    )

    with patch_bindings(mock_B):
        try:
            yield mock_B
        finally:
            # Drain the cache *inside* the patch. The autouse _clean_cache
            # fixture also clears it, but fixture teardown runs in reverse
            # setup order, so that clear happens after this patch is lifted --
            # which would leave mock-built filesystems to be finalised against
            # the real bindings. DFSFileSystem's finaliser now refuses to tear
            # down handles through a different bindings module, so that is no
            # longer fatal, but leaking here instead of closing would quietly
            # weaken every assertion below about connection counts.
            _fs_cache.clear_fs_cache(close=True)


@pytest.fixture(autouse=True)
def _clean_cache():
    """A populated cache must not leak between tests."""
    _fs_cache.clear_fs_cache()
    yield
    _fs_cache.clear_fs_cache()


class TestCaching:
    def test_same_arguments_return_the_same_instance(self, mock_bindings):
        """The whole point: one connection, not one per call.

        Asserts pool_connect was called exactly once. Identity alone would also
        hold if the cache were bypassed and both calls happened to be compared
        wrongly, so the construction count is the real assertion.
        """
        a = _fs_cache.get_fs("p", "c")
        b = _fs_cache.get_fs("p", "c")

        assert a is b
        assert mock_bindings.pool_connect.call_count == 1, "connected more than once"

    def test_different_container_gets_its_own_connection(self, mock_bindings):
        a = _fs_cache.get_fs("p", "c1")
        b = _fs_cache.get_fs("p", "c2")

        assert a is not b
        assert mock_bindings.pool_connect.call_count == 2

    def test_different_kwargs_get_their_own_connection(self, mock_bindings):
        """readonly=True must not be served from a read-write cache entry."""
        rw = _fs_cache.get_fs("p", "c")
        ro = _fs_cache.get_fs("p", "c", readonly=True)

        assert rw is not ro
        assert mock_bindings.pool_connect.call_count == 2

    def test_kwargs_order_does_not_create_a_second_entry(self, mock_bindings):
        """Keys are built from sorted kwargs, so call order is irrelevant."""
        a = _fs_cache.get_fs("p", "c", readonly=False, retry_count=2)
        b = _fs_cache.get_fs("p", "c", retry_count=2, readonly=False)

        assert a is b
        assert mock_bindings.pool_connect.call_count == 1

    def test_unhashable_kwargs_raise_a_clear_error(self, mock_bindings):
        """An unhashable kwarg cannot form a cache key; say so explicitly.

        Letting the raw TypeError from hash() escape would point at the cache
        internals rather than the caller's argument.
        """
        with pytest.raises(TypeError, match="must be hashable"):
            _fs_cache.get_fs("p", "c", weird={"a": 1})


class TestClosedEntries:
    def test_a_closed_cached_filesystem_is_replaced(self, mock_bindings):
        """Closing one caller's handle must not poison every later get_fs().

        Without the is_alive check the cache would keep returning the closed
        instance and every subsequent user would get "DFSFileSystem is closed"
        for the life of the process.
        """
        first = _fs_cache.get_fs("p", "c")
        first.close()
        assert first.is_alive is False

        second = _fs_cache.get_fs("p", "c")

        assert second is not first
        assert second.is_alive is True
        assert mock_bindings.pool_connect.call_count == 2, "did not reconnect"

    def test_clear_fs_cache_reports_and_drops_entries(self, mock_bindings):
        _fs_cache.get_fs("p", "c1")
        _fs_cache.get_fs("p", "c2")

        removed = _fs_cache.clear_fs_cache()

        assert removed == 2
        assert _fs_cache.cache_info()["entries"] == 0

    def test_clear_fs_cache_can_close_as_it_drops(self, mock_bindings):
        fs = _fs_cache.get_fs("p", "c")
        assert fs.is_alive is True

        _fs_cache.clear_fs_cache(close=True)

        assert fs.is_alive is False, "close=True did not close the filesystem"

    def test_clear_fs_cache_default_leaves_instances_open(self, mock_bindings):
        """Default must not close: callers may still hold references."""
        fs = _fs_cache.get_fs("p", "c")

        _fs_cache.clear_fs_cache()

        assert fs.is_alive is True, "cleared cache closed a filesystem it did not own"


class TestProcessIsolation:
    def test_a_different_pid_does_not_reuse_the_cached_handle(self, mock_bindings):
        """The property that matters most."""
        parent = _fs_cache.get_fs("p", "c")
        real_pid = os.getpid()

        with patch.object(_fs_cache.os, "getpid", return_value=real_pid + 1):
            child = _fs_cache.get_fs("p", "c")

        assert child is not parent, "child was handed the parent's handle"
        assert mock_bindings.pool_connect.call_count == 2

    def test_a_new_fork_generation_does_not_reuse_the_cached_handle(self, mock_bindings):
        """Covers a recycled pid after fork, where pid alone would collide."""
        parent = _fs_cache.get_fs("p", "c")

        with patch.object(_fork, "generation", return_value=_fork.generation() + 1):
            child = _fs_cache.get_fs("p", "c")

        assert child is not parent
        assert mock_bindings.pool_connect.call_count == 2

    def test_entries_from_other_processes_are_pruned(self, mock_bindings):
        """A child must not retain references to the parent's filesystems.

        After fork the child inherits the whole dict. Those entries can never
        be returned, but they keep poisoned objects alive and imply the child
        owns handles it must not tear down.
        """
        real_pid = os.getpid()
        with patch.object(_fs_cache.os, "getpid", return_value=real_pid + 1):
            _fs_cache.get_fs("p", "other-proc")
        assert _fs_cache.cache_info()["entries"] == 1

        # Back as "ourselves": the foreign entry must be dropped.
        _fs_cache.get_fs("p", "c")

        info = _fs_cache.cache_info()
        assert info["entries"] == 1, f"foreign entry not pruned: {info['keys']}"
        assert info["keys"][0]["container"] == "c"
        assert all(k["pid"] == real_pid for k in info["keys"])


class TestConcurrency:
    def test_racing_threads_produce_exactly_one_connection(self, mock_bindings):
        """Eight threads asking for the same filesystem must open one handle.

        This is the bug get_fs exists to prevent, in miniature: without the lock
        held across construction, every thread sees an empty cache and connects.
        Asserting the construction count is what detects it: asserting only
        that all threads got *some* filesystem would pass with eight.
        """
        results: list[object] = []
        errors: list[BaseException] = []
        ready = threading.Barrier(8, timeout=10)
        lock = threading.Lock()

        def worker():
            try:
                ready.wait()
                fs = _fs_cache.get_fs("p", "c")
                with lock:
                    results.append(fs)
            except BaseException as exc:  # noqa: BLE001 - recording
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
            assert not t.is_alive(), "get_fs deadlocked"

        assert not errors, f"get_fs raised under concurrency: {errors!r}"
        assert len(results) == 8
        assert len({id(r) for r in results}) == 1, "threads got different instances"
        assert mock_bindings.pool_connect.call_count == 1, (
            f"opened {mock_bindings.pool_connect.call_count} connections for one key"
        )


class TestReduceMessage:
    def test_pickling_still_raises(self, mock_bindings):
        """Decided 2026-10-14: DFSFileSystem does not gain reconnect-on-unpickle."""
        import pickle

        fs = _fs_cache.get_fs("p", "c")
        with pytest.raises(TypeError):
            pickle.dumps(fs)

    def test_the_error_names_get_fs(self, mock_bindings):
        """The message must name the sanctioned replacement, not just refuse.

        An error that says "cannot be pickled" and stops there is what drove
        every example author to invent their own pid-keyed dict.
        """
        import pickle

        fs = _fs_cache.get_fs("p", "c")
        with pytest.raises(TypeError) as excinfo:
            pickle.dumps(fs)

        msg = str(excinfo.value)
        assert "get_fs" in msg, "error message does not point at get_fs()"
        assert "spawn" in msg, "error message does not mention spawn"


@pytest.mark.live
class TestGetFsLive:
    """Against a real cluster, on both matrix versions."""

    def test_returns_a_working_filesystem(self, daos_pool, daos_container, dfs_fs):
        """One real connection, reused, that can actually do I/O."""
        from daos_fsspec import get_fs

        try:
            fs = get_fs(daos_pool, daos_container)
            again = get_fs(daos_pool, daos_container)
            assert fs is again

            path = f"{dfs_fs._test_dir}/get_fs-roundtrip.bin"
            payload = b"get_fs works" * 8
            fs.pipe_file(path, payload)
            assert fs.cat_file(path) == payload
        finally:
            # close=True: this test owns these connections, and leaving them
            # open would leak a handle per matrix run.
            _fs_cache.clear_fs_cache(close=True)

    def test_cache_info_reflects_a_real_connection(self, daos_pool, daos_container):
        from daos_fsspec import cache_info, get_fs

        try:
            get_fs(daos_pool, daos_container)
            info = cache_info()
            assert info["entries"] == 1
            assert info["pid"] == os.getpid()
            assert info["keys"][0]["container"] == daos_container
        finally:
            _fs_cache.clear_fs_cache(close=True)
