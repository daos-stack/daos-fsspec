# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for fork."""

from __future__ import annotations

import os
import threading
import time

import pytest

from daos_fsspec import _fork
from daos_fsspec.core import DFSFileSystem

# Python warns that forking a multi-threaded process may deadlock the child.
# That warning is describing the exact bug under test: deliberately provoking
# it is the point here.
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")

# waitpid() polling interval and the ceiling that distinguishes
# "raised promptly" from "hung": the whole point of the feature.
_POLL = 0.01
_CHILD_TIMEOUT = 20.0


def _run_in_fork(child_fn, timeout: float = _CHILD_TIMEOUT) -> int:
    """Run child_fn in a forked child; return its exit status."""
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            child_fn()
            code = 0
        except BaseException:
            code = 1
        finally:
            # _exit, not sys.exit: skip atexit hooks and pytest teardown,
            # which would try to tear down inherited DAOS handles.
            os._exit(code)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return os.waitstatus_to_exitcode(status)
        time.sleep(_POLL)

    os.kill(pid, 9)
    os.waitpid(pid, 0)
    raise TimeoutError(f"child did not exit within {timeout}s: it hung")


class _Registered:
    """Stand-in for a handle-holding object, so the registry can be tested
    without a live DAOS connection."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._fork_generation = _fork.generation()
        self._forked = False

    def _poison_after_fork(self) -> None:
        self._forked = True


def test_registered_object_is_poisoned_in_child():
    obj = _Registered()
    _fork.register(obj)

    def child():
        assert obj._forked is True, "fork hook did not poison the object"
        assert _fork.generation() != obj._fork_generation

    assert _run_in_fork(child) == 0
    # The parent must be untouched: it still owns working handles.
    assert obj._forked is False


def test_parent_generation_unchanged_by_child_fork():
    before = _fork.generation()
    assert _run_in_fork(lambda: None) == 0
    assert _fork.generation() == before


def test_check_fork_passes_in_the_creating_process():
    fs = DFSFileSystem.__new__(DFSFileSystem)
    fs._pid = os.getpid()
    fs._fork_generation = _fork.generation()
    fs._forked = False
    fs._check_fork()  # must not raise


def test_check_fork_raises_when_poisoned():
    fs = DFSFileSystem.__new__(DFSFileSystem)
    fs._pid = os.getpid()
    fs._fork_generation = _fork.generation()
    fs._forked = True

    with pytest.raises(RuntimeError) as exc:
        fs._check_fork()
    assert "fork()" in str(exc.value)


def test_check_fork_raises_on_generation_mismatch():
    """Belt and braces: even if the hook never ran the object's poison method,
    a stale generation is enough to refuse."""
    fs = DFSFileSystem.__new__(DFSFileSystem)
    fs._pid = os.getpid()
    fs._fork_generation = _fork.generation() - 1
    fs._forked = False

    with pytest.raises(RuntimeError):
        fs._check_fork()


def test_check_closed_reports_fork_not_closed():
    """A forked child also looks 'open'. The fork message is the useful one -
    'DFSFileSystem is closed' would send someone down the wrong path."""
    fs = DFSFileSystem.__new__(DFSFileSystem)
    fs._pid = os.getpid()
    fs._fork_generation = _fork.generation()
    fs._forked = True
    fs._closed = False

    with pytest.raises(RuntimeError) as exc:
        fs._check_closed()
    assert "fork" in str(exc.value).lower()
    assert "is closed" not in str(exc.value)


def test_error_message_names_the_fix():
    """The hang gave no traceback at all, so this message is the entire
    diagnostic. It has to say what to do."""
    msg = str(_fork.fork_error(os.getpid()))
    assert "spawn" in msg
    assert "multiprocessing_context" in msg
    assert str(os.getpid()) in msg


def test_fork_hook_is_registered():
    """_fork must have registered *its own* hook with the interpreter."""
    parent_generation = _fork.generation()

    def child():
        if _fork.generation() == parent_generation:
            raise AssertionError("the after-in-child hook did not run")

    assert _run_in_fork(child) == 0, (
        "the fork hook is not registered: the child's generation matched the "
        "parent's, so inherited handles would not be poisoned"
    )


@pytest.mark.live
def test_forked_child_raises_instead_of_hanging(dfs_fs):
    """forked child raises instead of hanging."""

    def child():
        try:
            dfs_fs.ls("/")
        except RuntimeError as exc:
            assert "fork" in str(exc).lower()
            return
        raise AssertionError("expected RuntimeError, operation succeeded")

    assert _run_in_fork(child) == 0


@pytest.mark.live
def test_forked_child_close_does_not_call_into_daos(dfs_fs):
    """forked child close does not call into daos."""

    def child():
        dfs_fs.close()  # must return, not hang or segfault
        assert dfs_fs._closed is True
        assert dfs_fs._dfs is None

    assert _run_in_fork(child) == 0


@pytest.mark.live
def test_parent_still_works_after_child_forked(dfs_fs):
    """Poisoning is per-process: the parent's handles stay valid."""
    assert _run_in_fork(lambda: None) == 0
    dfs_fs.ls("/")
    assert dfs_fs._forked is False


# ── get_fs()'s own lock ──────────────────────────────────────────────────────


class TestFsCacheLockSurvivesFork:
    """The one lock the fork hook used to miss.

    ``_fs_cache._lock`` is held across connection setup; deliberately, so
    two threads racing on one key produce a single connection rather than two
    with one leaked. That also makes it the widest window for a fork to land
    inside, and a child inheriting it locked blocks forever on its first
    ``get_fs()``: the holder is a thread that does not exist there.

    ``_fork`` replaces its own registry lock, and ``DFSFileSystem`` and
    ``DFSFileSystemPool`` replace theirs in ``_poison_after_fork``. This module
    was simply not registered; so the silent hang ``_fork`` exists to
    eliminate was reachable in the very API the fork error message recommends.

    Verified A/B before fixing: child hung without the guard, exited 0 with it.
    """

    @staticmethod
    def _mock_bindings():
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        class _FakeDaosError(Exception):
            is_retryable = False

        mock_B = MagicMock()
        mock_B._HAS_DFS_CONNECT = False
        mock_B.DaosError = _FakeDaosError
        mock_B.RETRYABLE_ERRORS = ()
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(name="root"),
            SimpleNamespace(size=0, mode=0o040755),
        )
        return mock_B

    def test_get_fs_in_a_child_rejects_without_blocking_on_the_inherited_lock(self):
        """Fork while another thread holds the cache lock, then reject safely."""
        from conftest import patch_bindings

        from daos_fsspec import _fs_cache

        with patch_bindings(self._mock_bindings()):
            held = threading.Event()
            release = threading.Event()

            def holder():
                with _fs_cache._lock:
                    held.set()
                    release.wait(_CHILD_TIMEOUT)

            thread = threading.Thread(target=holder, daemon=True)
            thread.start()
            assert held.wait(timeout=5.0), "could not acquire the cache lock for the test"

            try:

                def child():
                    try:
                        _fs_cache.get_fs("p", "c")
                    except RuntimeError as exc:
                        assert "spawn" in str(exc)
                        return
                    raise AssertionError("get_fs constructed DAOS state in a forked child")

                # _run_in_fork times out if this reaches an inherited lock.
                code = _run_in_fork(child)
            finally:
                release.set()
                thread.join(timeout=5.0)
                _fs_cache.clear_fs_cache()

        assert code == 0, f"get_fs() did not reject cleanly in the child (exit {code})"

    def test_direct_construction_is_also_rejected_in_a_forked_child(self):
        """Bypassing get_fs must not bypass the process-global DAOS guard."""
        from conftest import patch_bindings

        from daos_fsspec.core import DFSFileSystem

        with patch_bindings(self._mock_bindings()):

            def child():
                try:
                    DFSFileSystem(pool="p", container="c")
                except RuntimeError as exc:
                    assert "spawn" in str(exc)
                    return
                raise AssertionError("DFSFileSystem initialized after fork")

            code = _run_in_fork(child)

        assert code == 0, f"constructor did not reject cleanly in the child (exit {code})"

    def test_pool_construction_is_rejected_before_allocating_connections(self):
        from conftest import patch_bindings

        from daos_fsspec.pool import DFSFileSystemPool

        bindings = self._mock_bindings()
        with patch_bindings(bindings):

            def child():
                try:
                    DFSFileSystemPool(pool="p", container="c", size=2)
                except RuntimeError as exc:
                    assert "spawn" in str(exc)
                    assert bindings.pool_connect.call_count == 0
                    return
                raise AssertionError("DFSFileSystemPool initialized after fork")

            code = _run_in_fork(child)

        assert code == 0, f"pool constructor did not reject cleanly (exit {code})"

    def test_the_child_does_not_inherit_the_parents_cache_entries(self):
        """Inherited entries can never be returned; holding them is a leak.

        They are keyed by the parent's pid and fork generation, so every one is
        unreachable. Dropping them in the child also releases the poisoned
        filesystems, whose finalisers then see the generation change and skip
        libdfs rather than releasing handles the parent still owns.
        """
        from conftest import patch_bindings

        from daos_fsspec import _fs_cache

        with patch_bindings(self._mock_bindings()):
            _fs_cache.get_fs("p", "parent-entry")
            assert _fs_cache.cache_info()["entries"] == 1

            def child():
                if _fs_cache.cache_info()["entries"] != 0:
                    raise AssertionError("child inherited the parent's cache entries")

            try:
                code = _run_in_fork(child)
            finally:
                _fs_cache.clear_fs_cache()

        assert code == 0, "the child still held the parent's cache entries"

    def test_the_parent_keeps_its_cache_after_a_child_forks(self):
        """The guard runs only in the child; the parent must be untouched."""
        from conftest import patch_bindings

        from daos_fsspec import _fs_cache

        with patch_bindings(self._mock_bindings()):
            fs = _fs_cache.get_fs("p", "c")
            try:
                code = _run_in_fork(lambda: None)
                assert code == 0

                assert _fs_cache.cache_info()["entries"] == 1, (
                    "forking a child cleared the parent's cache"
                )
                assert _fs_cache.get_fs("p", "c") is fs, "parent lost its cached filesystem"
            finally:
                _fs_cache.clear_fs_cache(close=True)
