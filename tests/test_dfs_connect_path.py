# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for dfs connect path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings


class _FakeDaosError(Exception):
    is_retryable = False


@pytest.fixture
def connect_mock():
    """Bindings that advertise dfs_connect and record what gets called."""
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = True
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    # A real ctypes object, not a mock: _connect_via_dfs_connect passes this
    # to ctypes.byref(), which rejects anything else with a TypeError before
    # the code under test runs.
    from daos_fsspec._bindings import dfs_p as real_dfs_p

    mock_B.dfs_p.side_effect = real_dfs_p
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755, is_dir=True, type_name="directory"),
    )

    libdfs = MagicMock(name="libdfs")
    libdfs.dfs_connect.return_value = 0
    mock_B._ensure_libs.return_value = (MagicMock(name="libdaos"), libdfs)
    mock_B._check.return_value = None
    mock_B.libdfs = libdfs  # convenience handle for assertions

    with patch_bindings(mock_B):
        yield mock_B, libdfs


def _build(mock_B):
    from daos_fsspec.core import DFSFileSystem

    return DFSFileSystem(pool="mypool", container="mycont")


class TestConnect:
    def test_it_uses_dfs_connect_rather_than_the_four_step_path(self, connect_mock):
        mock_B, libdfs = connect_mock

        fs = _build(mock_B)
        try:
            assert libdfs.dfs_connect.call_count == 1, "dfs_connect was not used"
            assert mock_B.pool_connect.call_count == 0, "took the low-level path as well"
            assert mock_B.cont_open.call_count == 0
            assert mock_B.dfs_mount_fs.call_count == 0
            assert fs._uses_dfs_connect is True
        finally:
            fs._closed = True

    def test_dfs_init_is_called_before_dfs_connect(self, connect_mock):
        """dfs_connect() requires dfs_init(); the order is the contract."""
        mock_B, libdfs = connect_mock

        def connect(*args):
            mock_B._dfs_init.assert_called_once_with()
            return 0

        libdfs.dfs_connect.side_effect = connect
        fs = _build(mock_B)
        try:
            libdfs.dfs_connect.assert_called_once()
        finally:
            fs.close()

    def test_the_pool_and_container_labels_are_passed(self, connect_mock):
        mock_B, libdfs = connect_mock

        fs = _build(mock_B)
        try:
            args = libdfs.dfs_connect.call_args.args
            assert args[0] == b"mypool", f"pool label not forwarded: {args[0]!r}"
            assert args[2] == b"mycont", f"container label not forwarded: {args[2]!r}"
        finally:
            fs._closed = True

    def test_poh_and_coh_are_left_unset(self, connect_mock):
        """dfs_connect does not expose them; teardown must not assume it does.

        This is the difference between the two paths that teardown and
        reconnect branch on, so it is worth pinning rather than inferring.
        """
        mock_B, _ = connect_mock

        fs = _build(mock_B)
        try:
            assert fs._poh is None
            assert fs._coh is None
            assert fs._dfs is not None
            assert fs._root is not None
        finally:
            fs._closed = True

    def test_a_failed_root_lookup_disconnects_rather_than_leaking(self, connect_mock):
        """The error path inside the branch, which no test reached before."""
        mock_B, libdfs = connect_mock
        mock_B.dfs_lookup_path.side_effect = _FakeDaosError("root lookup failed")

        with pytest.raises(_FakeDaosError):
            _build(mock_B)

        assert libdfs.dfs_disconnect.call_count == 1, (
            "a failed root lookup left the dfs_connect handle open"
        )


class TestTeardown:
    def test_close_uses_dfs_disconnect(self, connect_mock):
        """Not umount/cont_close/pool_disconnect, which would be wrong here."""
        mock_B, libdfs = connect_mock
        fs = _build(mock_B)

        fs.close()

        assert libdfs.dfs_disconnect.call_count == 1, "close() did not disconnect"
        assert mock_B.dfs_umount.call_count == 0, "umounted a dfs_connect mount"
        assert mock_B.cont_close.call_count == 0
        assert mock_B.pool_disconnect.call_count == 0

    def test_the_finalizer_also_takes_the_dfs_connect_branch(self, connect_mock):
        mock_B, libdfs = connect_mock
        fs = _build(mock_B)

        assert fs._close_from_finalizer() is True
        assert libdfs.dfs_disconnect.call_count == 1
        assert mock_B.dfs_umount.call_count == 0


class TestReconnect:
    def test_reconnect_disconnects_then_connects_again(self, connect_mock):
        mock_B, libdfs = connect_mock
        fs = _build(mock_B)
        try:
            first_generation = fs._mount_generation

            fs._reconnect()

            assert libdfs.dfs_disconnect.call_count == 1, "old mount not disconnected"
            assert libdfs.dfs_connect.call_count == 2, "did not reconnect"
            assert fs._mount_generation != first_generation, "generation did not advance"
            assert fs._closed is False
        finally:
            fs._closed = True

    def test_a_reconnect_keeps_using_the_same_path(self, connect_mock):
        """It must not silently switch to the four-step path mid-life."""
        mock_B, _ = connect_mock
        fs = _build(mock_B)
        try:
            fs._reconnect()

            assert fs._uses_dfs_connect is True
            assert mock_B.dfs_mount_fs.call_count == 0, "reconnect switched code paths"
        finally:
            fs._closed = True
