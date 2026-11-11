# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Failure contracts for transfers, reconnect and URI validation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_filesystem import _make_mock_fs_for_dfs_file


@pytest.fixture
def mocked_fs():
    fs, bindings, cleanup = _make_mock_fs_for_dfs_file()
    try:
        yield fs, bindings
    finally:
        cleanup()


def test_missing_upload_source_never_mutates_remote(mocked_fs, tmp_path):
    fs, b = mocked_fs
    with pytest.raises(FileNotFoundError):
        fs.put_file(str(tmp_path / "missing"), "/existing")
    b.dfs_open_file.assert_not_called()
    b.dfs_move.assert_not_called()


@pytest.mark.parametrize("failure", ["read", "write", "rename"])
def test_failed_upload_preserves_destination(mocked_fs, tmp_path, failure):
    fs, b = mocked_fs
    source = tmp_path / "source"
    source.write_bytes(b"replacement")
    remote = {"existing": b"original"}

    def create(dfs, parent, name, mode, flags, **kwargs):
        remote[name] = b""
        return name

    def write(dfs, obj, offset, data):
        if failure == "write":
            raise OSError("write failed")
        remote[obj] += data

    def move(dfs, parent, src, dst_parent, dst):
        raise OSError("rename failed")

    b.dfs_open_file.side_effect = create
    b.dfs_write.side_effect = write
    b.dfs_move.side_effect = move
    b.dfs_remove.side_effect = lambda dfs, parent, name, **kw: remote.pop(name)
    if failure == "read":
        with patch("builtins.open") as opened:
            opened.return_value.__enter__.return_value.read.side_effect = OSError("read failed")
            with pytest.raises(OSError):
                fs.put_file(str(source), "/existing")
    else:
        with pytest.raises(OSError):
            fs.put_file(str(source), "/existing")
    assert remote == {"existing": b"original"}


def test_upload_commits_after_close(mocked_fs, tmp_path):
    fs, b = mocked_fs
    source = tmp_path / "source"
    source.write_bytes(b"replacement")
    events = []
    b.dfs_write.side_effect = lambda *a: events.append("write")
    b.dfs_close.side_effect = lambda *a: events.append("close")
    b.dfs_move.side_effect = lambda *a: events.append("rename")
    fs.put_file(str(source), "/existing")
    assert events == ["write", "close", "rename"]
    assert b.dfs_move.call_args.args[-1] == "existing"


def test_early_eof_preserves_local_destination(mocked_fs, tmp_path):
    fs, b = mocked_fs
    destination = tmp_path / "existing"
    destination.write_bytes(b"original")
    b.dfs_lookup_path.return_value = (MagicMock(), SimpleNamespace(size=5, is_dir=False))
    b.dfs_read.side_effect = [b"ab", b""]
    with pytest.raises(OSError, match="premature EOF"):
        fs.get_file("/remote", str(destination))
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_short_reads_complete_before_local_replace(mocked_fs, tmp_path):
    fs, b = mocked_fs
    b.dfs_lookup_path.return_value = (MagicMock(), SimpleNamespace(size=5, is_dir=False))
    b.dfs_read.side_effect = [b"ab", b"cde"]
    destination = tmp_path / "out"
    fs.get_file("/remote", str(destination))
    assert destination.read_bytes() == b"abcde"


def test_unrelated_retry_preserves_open_writer(mocked_fs):
    fs, b = mocked_fs
    fs._retry_backoff = 0
    writer = fs.open("/buffered", "wb")
    writer.write(b"pending")
    generation = fs._mount_generation

    def fail():
        raise b.DaosError(b.DER_UNREACH, "injected")

    with pytest.raises(b.DaosError):
        fs._retry(fail)
    assert fs._mount_generation == generation
    assert not writer.closed
    b.dfs_umount.assert_not_called()
    writer.close()
    assert b.dfs_write.call_args.args[-1] == b"pending"


@pytest.mark.parametrize("operation", ["rm", "info", "open", "cat"])
def test_foreign_uri_rejected_before_native_calls(mocked_fs, operation):
    fs, b = mocked_fs
    b.reset_mock()
    with pytest.raises(ValueError, match="does not match"):
        getattr(fs, operation)("dfs://other/other/victim")
    b.dfs_remove.assert_not_called()
    b.dfs_lookup_path.assert_not_called()
    b.dfs_open_file.assert_not_called()


def test_bound_uri_and_class_parsing(mocked_fs):
    fs, _ = mocked_fs
    assert fs._strip_protocol("dfs://test-pool/test-cont/file") == "/file"
    assert type(fs)._strip_protocol("dfs://other/other/file") == "/file"


def test_utf8_byte_limits(mocked_fs):
    fs, _ = mocked_fs
    assert fs._validate_path("/" + "é" * 127)
    with pytest.raises(ValueError, match="component too long"):
        fs._validate_path("/" + "é" * 128)
    with pytest.raises(ValueError, match="path too long"):
        fs._validate_path(("/" + "é" * 100) * 21)


def test_forced_teardown_reports_discarded_writes(mocked_fs):
    fs, _ = mocked_fs
    writer = fs.open("/buffered", "wb")
    writer.write(b"pending")
    fs.close()
    with pytest.raises(OSError, match="discarded"):
        writer.close()


@pytest.mark.live
def test_live_upload_failure_preserves_existing_file(dfs_fs, tmp_path):
    from daos_fsspec import _bindings

    remote = dfs_fs._test_dir + "/existing"
    dfs_fs.pipe_file(remote, b"original")
    with pytest.raises(FileNotFoundError):
        dfs_fs.put_file(str(tmp_path / "missing"), remote)
    source = tmp_path / "source"
    source.write_bytes(b"replacement")
    with patch.object(_bindings, "dfs_write", side_effect=OSError("injected write failure")):
        with pytest.raises(OSError, match="injected"):
            dfs_fs.put_file(str(source), remote)
    assert dfs_fs.cat_file(remote) == b"original"
    assert dfs_fs.ls(dfs_fs._test_dir, detail=False) == [remote]


@pytest.mark.live
def test_live_transfer_replaces_existing_destinations(dfs_fs, tmp_path):
    remote = dfs_fs._test_dir + "/existing"
    dfs_fs.pipe_file(remote, b"old remote")
    source = tmp_path / "source"
    source.write_bytes(b"replacement")
    dfs_fs.put_file(str(source), remote)
    assert dfs_fs.cat_file(remote) == b"replacement"
    destination = tmp_path / "destination"
    destination.write_bytes(b"old local")
    dfs_fs.get_file(remote, str(destination))
    assert destination.read_bytes() == b"replacement"
    assert dfs_fs.ls(dfs_fs._test_dir, detail=False) == [remote]
