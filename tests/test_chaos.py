# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for chaos."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import patch_bindings

# ── offline: retry driven by the real DaosError ──────────────────────────────
#
# These were live-gated, which meant this file contributed *nothing* to CI --
# all twelve tests needed a cluster. Their distinctive value is that they drive
# _retry with the real B.DaosError and the real B.RETRYABLE_ERRORS, which no
# other offline test does: every mocked fixture substitutes its own exception
# class and its own retryable set, so they exercise the mechanism against their
# own data. A cluster adds nothing to that, so the connection is mocked and the
# error machinery is left real.


@pytest.fixture
def retry_fs():
    """A filesystem with mocked libdfs but the genuine error types."""
    from daos_fsspec import _bindings as real_B

    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = real_B.DaosError
    mock_B.RETRYABLE_ERRORS = real_B.RETRYABLE_ERRORS
    mock_B.DER_UNREACH = real_B.DER_UNREACH
    mock_B.DER_HG = real_B.DER_HG
    mock_B.DER_TIMEDOUT = real_B.DER_TIMEDOUT
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755),
    )

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="p", container="c", retry_backoff=0.001)
        yield fs
        fs._closed = True


class TestRetryWithTheRealErrorTypes:
    """_retry against B.DaosError itself, not a stand-in."""

    def test_retry_on_daos_error(self, retry_fs):
        from daos_fsspec._bindings import DER_UNREACH, DaosError

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise DaosError(DER_UNREACH, "simulated transient error")
            return "success"

        with patch.object(retry_fs, "_reconnect"):
            assert retry_fs._retry(flaky) == "success"
        assert calls["n"] == 3, "the retry did not happen"

    def test_retry_exhaustion_raises(self, retry_fs):
        from daos_fsspec._bindings import DER_UNREACH, DaosError

        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise DaosError(DER_UNREACH, "permanent failure")

        with patch.object(retry_fs, "_reconnect"), pytest.raises(DaosError):
            retry_fs._retry(always_fail)

        assert calls["n"] == retry_fs._retry_count + 1, (
            f"attempted {calls['n']} times, expected {retry_fs._retry_count + 1}"
        )

    def test_no_retry_on_non_transient_error(self, retry_fs):
        """DER_NONEXIST is not transient; retrying would mask a real answer."""
        from daos_fsspec._bindings import DER_NONEXIST, DaosError

        calls = {"n": 0}

        def missing():
            calls["n"] += 1
            raise DaosError(DER_NONEXIST, "no such entry")

        with pytest.raises(DaosError):
            retry_fs._retry(missing)

        assert calls["n"] == 1, f"a non-transient error was retried {calls['n']} times"


class TestPoolExhaustionOffline:
    """The pool's notify-on-release, which was the only test of it anywhere.

    It needed a cluster to run, so CI never exercised the condition-variable
    handoff that makes a blocked acquire() wake up.
    """

    def test_release_wakes_a_blocked_acquirer(self):
        mock_B = MagicMock()
        mock_B._HAS_DFS_CONNECT = False
        mock_B.RETRYABLE_ERRORS = ()
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(),
            SimpleNamespace(size=0, mode=0o040755),
        )

        with patch_bindings(mock_B):
            from daos_fsspec.pool import DFSFileSystemPool

            pool = DFSFileSystemPool(pool="p", container="c", size=1)
            try:
                held = pool.acquire()
                acquired = threading.Event()

                def waiter():
                    fs = pool.acquire(timeout=10)
                    acquired.set()
                    pool.release(fs)

                thread = threading.Thread(target=waiter, daemon=True)
                thread.start()

                assert not acquired.wait(timeout=0.3), "acquire() returned from an exhausted pool"

                released_at = time.monotonic()
                pool.release(held)

                assert acquired.wait(timeout=10), (
                    "release() did not hand the connection over at all"
                )
                handoff = time.monotonic() - released_at

                # Promptness is the real assertion. Without notify(), the
                # waiter still gets there; acquire() re-checks the queue when
                # its own wait expires; so "did it eventually acquire?" is
                # satisfied by the timeout path and took 10s to pass. A
                # notified handoff is sub-millisecond; 2s is generous enough
                # not to be timing-sensitive and far below any poll interval.
                assert handoff < 2.0, (
                    f"the handoff took {handoff:.2f}s; release() is not "
                    "notifying the condition variable, so a blocked acquirer "
                    "only proceeds when its own wait times out"
                )
                thread.join(timeout=5)
            finally:
                pool.close(force=True)


@pytest.mark.live
class TestChaosTransientErrors:
    """Connection state against real handles.

    The retry tests that used to live here are offline now (see
    TestRetryWithTheRealErrorTypes): they drive _retry with a pure-Python
    callable, so a cluster added nothing. What is left genuinely needs real
    handles.
    """

    def test_health_check_returns_true(self, dfs_fs):
        """Health check should return True on healthy connection."""
        assert dfs_fs.health_check() is True

    def test_is_alive_on_open_connection(self, dfs_fs):
        """is_alive should return True on open connection."""
        assert dfs_fs.is_alive is True

    def test_is_alive_after_close(self, daos_pool, daos_container):
        """is_alive should return False after close."""
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool=daos_pool, container=daos_container)
        assert fs.is_alive is True
        fs.close()
        assert fs.is_alive is False


@pytest.mark.live
class TestChaosConnectionPool:
    """Test connection pool under stress."""

    def test_pool_acquire_release(self, daos_pool, daos_container):
        """Acquire and release connections from pool."""
        from daos_fsspec.pool import DFSFileSystemPool

        with DFSFileSystemPool(pool=daos_pool, container=daos_container, size=2) as pool:
            fs1 = pool.acquire()
            fs2 = pool.acquire()

            # Both should work
            assert fs1.health_check() is True
            assert fs2.health_check() is True

            pool.release(fs1)
            pool.release(fs2)

            # Pool should have 2 available
            status = pool.status()
            assert status["available"] == 2

    def test_pool_exhaustion_blocks(self, daos_pool, daos_container):
        """Pool should block when all connections are in use."""
        import threading

        from daos_fsspec.pool import DFSFileSystemPool

        with DFSFileSystemPool(pool=daos_pool, container=daos_container, size=1) as pool:
            fs = pool.acquire()

            acquired = threading.Event()

            def try_acquire():
                f = pool.acquire()
                acquired.set()
                pool.release(f)

            t = threading.Thread(target=try_acquire)
            t.start()

            # Should not acquire immediately (pool is empty)
            assert not acquired.wait(timeout=0.5)

            # Release and then it should acquire
            pool.release(fs)
            assert acquired.wait(timeout=5.0)
            t.join()


@pytest.mark.live
class TestChaosEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_file_operations(self, dfs_fs):
        """Operations on empty files should work."""
        d = dfs_fs._test_dir
        path = f"{d}/empty-chaos.bin"

        dfs_fs.pipe_file(path, b"")
        assert dfs_fs.exists(path)
        info = dfs_fs.info(path)
        assert info["size"] == 0
        assert dfs_fs.cat(path) == b""

    def test_rapid_create_delete_cycles(self, dfs_fs):
        """Rapid create/delete must stay correct and not accumulate handles.

        The name used to promise more than the body delivered: it asserted
        existence either side of each cycle and nothing at all about handles.
        It still cannot count libdfs handles; nothing exposes them; but it
        can check the filesystem is still usable afterwards, which a handle
        exhaustion would break, and that the final state is clean.
        """
        d = dfs_fs._test_dir
        path = f"{d}/rapid-chaos.txt"

        for i in range(50):
            dfs_fs.pipe_file(path, f"cycle-{i}".encode())
            assert dfs_fs.exists(path)
            dfs_fs.rm(path)
            assert not dfs_fs.exists(path)

        # Still usable after 50 cycles. Handle exhaustion shows up here rather
        # than in the loop, which only ever touched one path.
        probe = f"{d}/rapid-chaos-after.txt"
        dfs_fs.pipe_file(probe, b"still working")
        assert dfs_fs.cat(probe) == b"still working"
        assert dfs_fs.ls(d, detail=False), "the directory listing came back empty"
        dfs_fs.rm(probe)

    def test_concurrent_mkdir_same_path(self, dfs_fs):
        """Concurrent mkdir on same path should not crash."""
        import concurrent.futures

        d = dfs_fs._test_dir
        path = f"{d}/concurrent-mkdir"

        def try_mkdir():

            try:
                dfs_fs.mkdir(path)
                return "ok"
            except FileExistsError:
                return "exists"
            except Exception as e:  # noqa: BLE001  # Surface the wrong error type.
                return f"wrong-error:{type(e).__name__}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(try_mkdir) for _ in range(4)]
            results = [f.result() for f in futures]

        # Every outcome must be a clean success/already-exists: no other error.
        assert all(r in ("ok", "exists") for r in results), results
        # Post-fault state must be consistent: the path exists exactly once and
        # is a directory (idempotent concurrent mkdir, not a corrupted entry).
        assert dfs_fs.info(path)["type"] == "directory"
        parent = path.rsplit("/", 1)[0] or "/"
        base = path.rsplit("/", 1)[-1]
        basenames = [
            e["name"].rstrip("/").rsplit("/", 1)[-1] for e in dfs_fs.ls(parent, detail=True)
        ]
        assert basenames.count(base) == 1, f"duplicate dir entries: {basenames}"

    def test_special_characters_in_names(self, dfs_fs):
        """Files with special characters should work."""
        d = dfs_fs._test_dir
        names = [
            "file with spaces.txt",
            "file.with.dots.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "MiXeD.CaSe.TXT",
        ]

        for name in names:
            path = f"{d}/{name}"
            data = f"content-{name}".encode()
            dfs_fs.pipe_file(path, data)
            assert dfs_fs.cat(path) == data
