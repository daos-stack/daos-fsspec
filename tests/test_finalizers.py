# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for finalizers."""

from __future__ import annotations

import gc
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings

# A finaliser must settle in well under this. It is a hang detector, not a
# performance assertion.
_NO_BLOCK_DEADLINE = 5.0


class _FakeDaosError(Exception):
    is_retryable = False


def _mock_bindings() -> MagicMock:
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
    return mock_B


def _teardown_calls(mock_B) -> dict[str, int]:
    """How many times each teardown entry point was invoked."""
    return {
        name: getattr(mock_B, name).call_count
        for name in ("dfs_release", "dfs_umount", "cont_close", "pool_disconnect")
    }


def _ran_without_blocking(fn, deadline: float = _NO_BLOCK_DEADLINE):
    """Call fn on a daemon thread; fail if it does not return in time."""
    box: dict[str, object] = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=deadline)
    if thread.is_alive():
        pytest.fail(
            f"finaliser did not return within {deadline:.0f}s; it blocked. "
            "A finaliser runs on whichever thread triggered collection, so "
            "blocking here stalls unrelated code and can hang at shutdown."
        )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result")


@pytest.fixture
def fs_and_mock():
    """A connected DFSFileSystem plus its mocked bindings, counters zeroed."""
    mock_B = _mock_bindings()
    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="p", container="c")
        for name in ("dfs_release", "dfs_umount", "cont_close", "pool_disconnect"):
            getattr(mock_B, name).reset_mock()
        yield fs, mock_B
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass


class TestFilesystemFinalizerReleases:
    """The half that fixes a leak."""

    def test_dropping_without_close_releases_the_handles(self, fs_and_mock):
        """The whole point: no close() must not mean handles held until exit."""
        fs, mock_B = fs_and_mock

        released = fs._close_from_finalizer()

        assert released is True, "finaliser declined to release an idle filesystem"
        calls = _teardown_calls(mock_B)
        assert calls["dfs_release"] == 1, f"root handle not released: {calls}"
        assert calls["dfs_umount"] == 1, f"mount not umounted: {calls}"
        assert calls["cont_close"] == 1, f"container not closed: {calls}"
        assert calls["pool_disconnect"] == 1, f"pool not disconnected: {calls}"

    def test_a_real_del_runs_the_release(self):
        """Proves __del__ is wired to it, not just that the helper works."""
        mock_B = _mock_bindings()
        with patch_bindings(mock_B):
            from daos_fsspec.core import DFSFileSystem

            def build_and_drop():
                fs = DFSFileSystem(pool="p", container="c")
                for name in ("dfs_umount", "cont_close", "pool_disconnect"):
                    getattr(mock_B, name).reset_mock()
                del fs

            build_and_drop()
            gc.collect()

            calls = _teardown_calls(mock_B)

        assert calls["dfs_umount"] == 1, f"__del__ did not umount: {calls}"
        assert calls["pool_disconnect"] == 1, f"__del__ did not disconnect: {calls}"

    def test_finalizer_after_an_explicit_close_does_nothing(self, fs_and_mock):
        """Double teardown would be a double free."""
        fs, mock_B = fs_and_mock
        fs.close()
        before = _teardown_calls(mock_B)

        assert fs._close_from_finalizer() is False
        assert _teardown_calls(mock_B) == before, "finaliser released already-freed handles"

    def test_close_is_still_idempotent(self, fs_and_mock):
        fs, mock_B = fs_and_mock
        fs.close()
        after_first = _teardown_calls(mock_B)
        fs.close()

        assert _teardown_calls(mock_B) == after_first, "second close() released handles again"


class TestFilesystemFinalizerNeverBlocks:
    """The half that matters more."""

    def test_it_gives_up_when_another_thread_holds_the_lock(self, fs_and_mock):
        """Teardown in flight elsewhere: leak, do not wait.

        The old pool finaliser would have blocked here until the holder let go,
        on a thread that never asked to be involved.
        """
        fs, mock_B = fs_and_mock
        holding = threading.Event()
        release = threading.Event()

        def hold_exclusive():
            with fs._lock.write():
                holding.set()
                release.wait(timeout=_NO_BLOCK_DEADLINE)

        holder = threading.Thread(target=hold_exclusive, daemon=True)
        holder.start()
        assert holding.wait(timeout=_NO_BLOCK_DEADLINE), "could not set up the contended lock"

        try:
            result = _ran_without_blocking(fs._close_from_finalizer)
            assert result is False, "finaliser tore down a filesystem it could not lock"
            assert _teardown_calls(mock_B) == dict.fromkeys(
                ("dfs_release", "dfs_umount", "cont_close", "pool_disconnect"), 0
            ), "handles were released without the exclusive lock"
        finally:
            release.set()
            holder.join(timeout=_NO_BLOCK_DEADLINE)

    def test_it_does_not_deadlock_when_collection_fires_under_a_read_lock(self, fs_and_mock):
        """it does not deadlock when collection fires under a read lock."""
        fs, mock_B = fs_and_mock

        # Called on *this* thread, deliberately. RWLock read depth is
        # thread-local, so running the finaliser on a helper thread took the
        # ordinary contended path and never reached the lock-upgrade branch
        # this test documents; deleting the `except RuntimeError` handler
        # left it passing, and it was a duplicate of the test above.
        #
        # Safe to call directly: acquire_write() refuses the upgrade
        # immediately rather than waiting, so there is nothing to time out.
        #
        # This pins the behaviour, not which handler produces it. The upgrade
        # raises RuntimeError and _close_from_finalizer also has a catch-all,
        # so removing the specific `except RuntimeError` still returns False --
        # correctly. What matters is that the finaliser declines and does not
        # umount a mount in use, which is what is asserted.
        with fs._lock.read():
            result = fs._close_from_finalizer()

        assert result is False, "finaliser upgraded the lock from inside a read section"
        assert _teardown_calls(mock_B)["dfs_umount"] == 0, "umounted a mount still in use"

    def test_it_skips_libdfs_entirely_after_a_fork(self, fs_and_mock):
        """Inherited handles are unusable and calling in would hang."""
        fs, mock_B = fs_and_mock
        fs._forked = True

        result = _ran_without_blocking(fs._close_from_finalizer)

        assert result is False
        assert _teardown_calls(mock_B) == dict.fromkeys(
            ("dfs_release", "dfs_umount", "cont_close", "pool_disconnect"), 0
        ), "called into libdfs with handles inherited across a fork"
        assert fs._dfs is None, "fork path did not drop the handle references"

    def test_it_refuses_to_tear_down_through_different_bindings(self, fs_and_mock):
        """A handle may only be released by the library that created it.

        The finaliser can run long after a bindings swap; a module reload, or
        a test whose ctypes patch has since been lifted. Handing a handle to a
        different library is undefined, and if the replacement is a mock,
        allocating its children during collection is fatal rather than merely
        wrong. This is not hypothetical: it segfaulted the live suite, where
        cached mock-built filesystems were finalised after the patch ended.
        """
        fs, mock_B = fs_and_mock
        import daos_fsspec.core as core

        original = core.B
        try:
            core.B = _mock_bindings()  # a *different* bindings object
            assert fs._close_from_finalizer() is False, (
                "tore down handles through bindings that did not create them"
            )
            assert _teardown_calls(mock_B) == dict.fromkeys(
                ("dfs_release", "dfs_umount", "cont_close", "pool_disconnect"), 0
            )
        finally:
            core.B = original

    def test_it_survives_the_bindings_being_torn_down(self, fs_and_mock):
        """Interpreter shutdown: B may be partly collected."""
        fs, _ = fs_and_mock
        import daos_fsspec.core as core

        original = core.B
        try:
            core.B = SimpleNamespace()  # no dfs_release attribute
            assert _ran_without_blocking(fs._close_from_finalizer) is False
        finally:
            core.B = original


class TestPoolFinalizer:
    """DFSFileSystemPool.__del__ used to take two blocking locks."""

    @staticmethod
    def _pool(size: int = 3):
        from daos_fsspec.pool import DFSFileSystemPool

        return DFSFileSystemPool(pool="p", container="c", size=size)

    def test_it_closes_every_pooled_connection(self):
        mock_B = _mock_bindings()
        with patch_bindings(mock_B):
            pool = self._pool(3)
            connections = list(pool._all)
            for name in ("dfs_umount", "cont_close", "pool_disconnect"):
                getattr(mock_B, name).reset_mock()

            pool._close_from_finalizer()

            assert all(fs._closed for fs in connections), "a pooled connection stayed open"
            assert _teardown_calls(mock_B)["dfs_umount"] == 3, (
                "expected one umount per pooled connection"
            )
            assert not pool._all and not pool._in_use

    def test_it_gives_up_rather_than_waiting_for_the_pool_lock(self):
        """Another thread mid-acquire/release must not stall the finaliser."""
        mock_B = _mock_bindings()
        with patch_bindings(mock_B):
            pool = self._pool(2)
            for name in ("dfs_umount", "cont_close", "pool_disconnect"):
                getattr(mock_B, name).reset_mock()

            holding = threading.Event()
            release = threading.Event()

            def hold_pool_lock():
                with pool._lock:
                    holding.set()
                    release.wait(timeout=_NO_BLOCK_DEADLINE)

            holder = threading.Thread(target=hold_pool_lock, daemon=True)
            holder.start()
            assert holding.wait(timeout=_NO_BLOCK_DEADLINE), "could not contend the pool lock"

            try:
                _ran_without_blocking(pool._close_from_finalizer)
                assert _teardown_calls(mock_B)["dfs_umount"] == 0, (
                    "tore down connections without the pool lock"
                )
            finally:
                release.set()
                holder.join(timeout=_NO_BLOCK_DEADLINE)
                pool.close(force=True)

    def test_it_skips_everything_after_a_fork(self):
        """No libdfs call at all in a forked child; it would hang.

        Poisons the connections as well as the pool, because that is what
        actually happens: both classes register with _fork, and the
        after-in-child hook calls _poison_after_fork() on each. Setting the
        flag on the pool alone is not a forked child, and it hides this test's
        own subtlety; clearing pool._all drops the last reference to every
        connection, so their finalisers run immediately. They skip only because
        they are poisoned too.
        """
        mock_B = _mock_bindings()
        with patch_bindings(mock_B):
            pool = self._pool(2)
            for name in ("dfs_umount", "cont_close", "pool_disconnect"):
                getattr(mock_B, name).reset_mock()
            for fs in pool._all:
                fs._poison_after_fork()
            pool._poison_after_fork()

            pool._close_from_finalizer()
            gc.collect()  # force the dropped connections' finalisers to run now

            assert _teardown_calls(mock_B)["dfs_umount"] == 0, (
                "closed connections inherited across a fork"
            )
            assert not pool._all, "fork path did not drop the connection references"

    def test_a_real_del_runs_the_pool_teardown(self):
        mock_B = _mock_bindings()
        with patch_bindings(mock_B):

            def build_and_drop():
                pool = self._pool(2)
                for name in ("dfs_umount", "cont_close", "pool_disconnect"):
                    getattr(mock_B, name).reset_mock()
                del pool

            build_and_drop()
            gc.collect()

            assert _teardown_calls(mock_B)["dfs_umount"] == 2, (
                "pool __del__ did not tear down its connections"
            )


class TestFinalizerPolicyIsUniform:
    """Guard the agreement itself, not just each implementation."""

    def test_all_three_owners_expose_a_non_blocking_finalizer_path(self):
        """A new handle owner must not quietly reintroduce a blocking __del__."""
        from daos_fsspec.core import DFSFileSystem
        from daos_fsspec.dfs_file import DFSFile
        from daos_fsspec.pool import DFSFileSystemPool

        assert hasattr(DFSFileSystem, "_close_from_finalizer")
        assert hasattr(DFSFileSystemPool, "_close_from_finalizer")
        # DFSFile keeps its own inline guards; it predates this and its
        # finaliser is the model the other two were changed to follow.
        assert hasattr(DFSFile, "__del__")

    def test_no_finalizer_waits_on_a_future_or_a_lock(self):
        """Static check: the words that mean "block" must not appear in __del__."""
        import ast
        import inspect

        from daos_fsspec import core, dfs_file, pool

        offenders = []
        for module in (core, dfs_file, pool):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                # Not just __del__: all three are thin wrappers that delegate,
                # so scanning only them policed the functions containing no
                # logic. Injecting executor.shutdown(wait=True) or an unbounded
                # acquire_write() into _close_from_finalizer passed happily.
                if not (
                    isinstance(node, ast.FunctionDef)
                    and node.name in ("__del__", "_close_from_finalizer")
                ):
                    continue

                statements = node.body
                if (
                    statements
                    and isinstance(statements[0], ast.Expr)
                    and isinstance(statements[0].value, ast.Constant)
                    and isinstance(statements[0].value.value, str)
                ):
                    statements = statements[1:]
                body = "\n".join(ast.unparse(stmt) for stmt in statements)
                # Strip the non-blocking forms first: they are precisely how a
                # finaliser is *supposed* to take a lock, and a substring
                # search would otherwise flag the fix as the defect.
                for safe in (
                    "acquire(blocking=False)",
                    "acquire_write(timeout=0)",
                    "acquire_read(blocking=False)",
                ):
                    body = body.replace(safe, "<non-blocking acquire>")
                for bad in (
                    "wait=True",
                    ".join(",
                    ".result(",
                    "acquire()",
                    "acquire_write()",
                    "_lock.write()",
                ):
                    if bad in body:
                        offenders.append(f"{module.__name__}.__del__ contains {bad!r}")

        assert not offenders, "finalisers must not block: " + "; ".join(offenders)
