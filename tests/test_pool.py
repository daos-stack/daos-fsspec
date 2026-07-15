# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DFSFileSystemPool (no DAOS needed)."""

from unittest.mock import MagicMock, patch

import pytest

from daos_fsspec.pool import DFSFileSystemPool


class TestPoolInit:
    """Pool initialization validation."""

    def test_size_must_be_positive(self):
        with pytest.raises(ValueError, match="pool size must be >= 1"):
            DFSFileSystemPool(pool="test", container="test", size=0)

    def test_has_expected_attributes(self):
        assert hasattr(DFSFileSystemPool, "acquire")
        assert hasattr(DFSFileSystemPool, "release")
        assert hasattr(DFSFileSystemPool, "connection")
        assert hasattr(DFSFileSystemPool, "close")
        assert hasattr(DFSFileSystemPool, "health_check")
        assert hasattr(DFSFileSystemPool, "status")


class TestPoolOperations:
    """Pool acquire/release/close — mock-based, no DAOS."""

    @patch("daos_fsspec.pool.DFSFileSystem")
    def _make_pool(self, mock_cls, size=2):
        mock_cls.return_value = MagicMock()
        pool = DFSFileSystemPool(pool="test-pool", container="test-cont", size=size)
        return pool

    def test_acquire_release(self):
        pool = self._make_pool()
        fs = pool.acquire()
        assert fs is not None
        pool.release(fs)
        pool.close(force=True)

    def test_acquire_all_then_block(self):
        pool = self._make_pool(size=2)
        fs1 = pool.acquire()
        fs2 = pool.acquire()

        # All connections in use — third acquire should timeout
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.05)

        pool.release(fs1)
        pool.release(fs2)
        pool.close(force=True)

    def test_acquire_timeout_raises(self):
        pool = self._make_pool(size=1)
        fs = pool.acquire()
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.01)
        pool.release(fs)
        pool.close(force=True)

    def test_acquire_after_close_raises(self):
        pool = self._make_pool()
        pool.close(force=True)
        with pytest.raises(RuntimeError, match="closed"):
            pool.acquire()

    def test_close_force_closes_in_use(self):
        pool = self._make_pool(size=1)
        fs = pool.acquire()
        pool.close(force=True)
        fs.close.assert_called_once()

    def test_close_default_defers_in_use(self):
        pool = self._make_pool(size=1)
        fs = pool.acquire()
        pool.close()  # force=False by default
        # In-use connection should NOT be closed yet
        fs.close.assert_not_called()
        # After release, it should be closed
        pool.release(fs)
        fs.close.assert_called_once()

    def test_context_manager(self):
        pool = self._make_pool()
        with pool.connection() as fs:
            assert fs is not None
        pool.close(force=True)

    def test_release_wrong_fs_ignored(self):
        pool = self._make_pool()
        fake_fs = MagicMock()
        pool.release(fake_fs)  # Should not raise
        pool.close(force=True)

    def test_double_release_ignored(self):
        pool = self._make_pool(size=1)
        fs = pool.acquire()
        pool.release(fs)
        pool.release(fs)  # Double release — should not add to queue twice
        s = pool.status()
        assert s["available"] == 1  # Not 2
        assert s["in_use"] == 0
        pool.close(force=True)

    def test_status(self):
        pool = self._make_pool(size=2)
        s = pool.status()
        assert s["pool"] == "test-pool"
        assert s["container"] == "test-cont"
        assert s["size"] == 2
        assert s["available"] == 2
        assert s["in_use"] == 0
        assert not s["closed"]
        pool.close(force=True)

    def test_status_after_acquire(self):
        pool = self._make_pool(size=2)
        fs = pool.acquire()
        s = pool.status()
        assert s["available"] == 1
        assert s["in_use"] == 1
        pool.release(fs)
        pool.close(force=True)

    def test_repr(self):
        pool = self._make_pool(size=3)
        r = repr(pool)
        assert "test-pool" in r
        assert "test-cont" in r
        assert "3" in r
        pool.close(force=True)

    def test_health_check(self):
        pool = self._make_pool(size=2)
        results = pool.health_check()
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "connection_0" in results
        assert "connection_1" in results
        pool.close(force=True)

    def test_pool_context_manager(self):
        """Pool __enter__/__exit__ closes all connections."""
        pool = self._make_pool(size=2)
        with pool:
            fs = pool.acquire()
            assert fs is not None
            pool.release(fs)
        assert pool._closed

    def test_release_dead_connection_not_returned(self):
        """Dead connections should not be returned to the pool."""
        pool = self._make_pool(size=1)
        fs = pool.acquire()

        # Simulate a dead connection
        fs.is_alive = False

        pool.release(fs)

        # Pool should have 0 available (dead connection discarded)
        s = pool.status()
        assert s["available"] == 0
        assert s["in_use"] == 0
        fs.close.assert_called_once()
        pool.close(force=True)
