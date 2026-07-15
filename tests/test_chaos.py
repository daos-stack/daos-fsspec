# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Chaos and fault injection tests for daos-fsspec.

These tests verify behavior under adverse conditions.
Requires a running DAOS cluster.
"""

import pytest


@pytest.mark.live
class TestChaosTransientErrors:
    """Test retry and reconnection behavior."""

    def test_retry_on_daos_error(self, dfs_fs):
        """Verify retry logic works for transient errors."""
        from daos_fsspec._bindings import DER_UNREACH, DaosError

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DaosError(DER_UNREACH, "simulated transient error")
            return "success"

        # The retry wrapper should succeed after retries
        result = dfs_fs._retry(flaky_operation)
        assert result == "success"
        assert call_count == 3

    def test_retry_exhaustion_raises(self, dfs_fs):
        """Verify retry raises after exhausting attempts."""
        from daos_fsspec._bindings import DER_UNREACH, DaosError

        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise DaosError(DER_UNREACH, "permanent failure")

        with pytest.raises(DaosError):
            dfs_fs._retry(always_fail)

        # Should have tried retry_count + 1 times
        assert call_count == dfs_fs._retry_count + 1

    def test_no_retry_on_non_transient_error(self, dfs_fs):
        """Verify non-transient errors are not retried."""
        from daos_fsspec._bindings import DER_NONEXIST, DaosError

        call_count = 0

        def not_found():
            nonlocal call_count
            call_count += 1
            raise DaosError(DER_NONEXIST, "not found")

        with pytest.raises(DaosError):
            dfs_fs._retry(not_found)

        # Should only try once (no retry for non-transient)
        assert call_count == 1

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

    def test_rapid_create_delete_cycles(self, dfs_fs, handle_tracker):
        """Rapid create/delete must not leak DFS object handles.

        Counts acquisitions vs. releases through the binding layer across the
        loop and asserts they balance — the persistent root handle predates the
        tracker, so a healthy run nets to zero.
        """
        d = dfs_fs._test_dir
        path = f"{d}/rapid-chaos.txt"

        with handle_tracker.track():
            for i in range(50):
                dfs_fs.pipe_file(path, f"cycle-{i}".encode())
                assert dfs_fs.exists(path)
                dfs_fs.rm(path)
                assert not dfs_fs.exists(path)

        # Sanity: the wrapped acquire path was actually exercised...
        assert handle_tracker.acquired > 0
        # ...and every handle acquired during the loop was released.
        assert handle_tracker.net == 0, (
            f"leaked {handle_tracker.net} DFS handles "
            f"({handle_tracker.acquired} acquired, {handle_tracker.released} released)"
        )

    def test_concurrent_mkdir_same_path(self, dfs_fs):
        """Concurrent mkdir on same path should not crash."""
        import concurrent.futures

        d = dfs_fs._test_dir
        path = f"{d}/concurrent-mkdir"

        def try_mkdir():
            # Return a tag so we verify the OUTCOME, not just "didn't crash".
            # mkdir(create_parents=True) is idempotent (mkdir -p semantics) and
            # serialized by the per-fs lock, so each call must succeed or report
            # an already-exists — never a different (swallowed) error type.
            try:
                dfs_fs.mkdir(path)
                return "ok"
            except FileExistsError:
                return "exists"
            except Exception as e:  # noqa: BLE001 — surface the wrong error type
                return f"wrong-error:{type(e).__name__}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(try_mkdir) for _ in range(4)]
            results = [f.result() for f in futures]

        # Every outcome must be a clean success/already-exists — no other error.
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
