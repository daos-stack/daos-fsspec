# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for argonne regressions."""

from __future__ import annotations

import errno
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import patch_bindings

from daos_fsspec import _bindings as B

# ── get_file() on a directory ─────────────────────────────────────────────


def _stat(mode: int, size: int = 0) -> SimpleNamespace:
    import stat as stat_mod

    return SimpleNamespace(
        size=size,
        mode=mode,
        uid=0,
        gid=0,
        nlink=1,
        mtime=0.0,
        atime=0.0,
        ctime=0.0,
        is_dir=stat_mod.S_ISDIR(mode),
        type_name="directory" if stat_mod.S_ISDIR(mode) else "file",
    )


@pytest.fixture
def mock_fs():
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B._CHUNK_SIZE = 1 << 24
    mock_B.DaosError = B.DaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.pool_connect.return_value = MagicMock(name="poh")
    mock_B.cont_open.return_value = MagicMock(name="coh")
    mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
    mock_B.dfs_lookup_path.return_value = (MagicMock(name="root"), _stat(0o040755))

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="p", container="c")
        mock_B.dfs_lookup_path.reset_mock()
        yield fs, mock_B
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass


def test_get_file_on_directory_creates_it_and_does_not_read(mock_fs, tmp_path):
    """The bug behind 'fs.get(recursive=True) raises EINVAL'."""
    fs, mock_B = mock_fs
    mock_B.dfs_lookup_path.return_value = (MagicMock(name="dirobj"), _stat(0o040755, size=4096))

    dest = tmp_path / "subdir"
    assert fs.get_file("/some/dir", str(dest)) is None

    assert dest.is_dir(), "directory was not created locally"
    mock_B.dfs_read.assert_not_called()
    mock_B.dfs_release.assert_called_once()


def test_get_file_on_regular_file_still_downloads(mock_fs, tmp_path):
    fs, mock_B = mock_fs
    payload = b"hello daos"
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="fileobj"),
        _stat(0o100644, size=len(payload)),
    )
    mock_B.dfs_read.return_value = payload

    dest = tmp_path / "out.bin"
    fs.get_file("/some/file", str(dest))

    assert dest.read_bytes() == payload
    mock_B.dfs_read.assert_called()


def test_get_file_does_not_pay_an_extra_roundtrip_for_the_guard(mock_fs, tmp_path):
    """get file does not pay an extra roundtrip for the guard."""
    fs, mock_B = mock_fs
    mock_B.dfs_lookup_path.return_value = (MagicMock(name="dirobj"), _stat(0o040755))

    fs.get_file("/some/dir", str(tmp_path / "d"))

    assert mock_B.dfs_lookup_path.call_count == 1


# ── readdirplus aborting on a concurrent unlink ───────────────────────────


def _dirent_batch(names, array):
    for i, name in enumerate(names):
        array[i].d_name = name.encode() + b"\x00"


def test_readdirplus_falls_back_when_an_entry_vanishes():
    """The bug behind 'never list a directory another rank writes to'.

    DAOS readdir_int() stats each enumerated dkey and aborts the whole batch
    on the first failure, so one peer deleting one file turned an entire
    listing into ENOENT.
    """
    libdfs = MagicMock()
    libdfs.dfs_readdirplus.return_value = errno.ENOENT

    def fake_readdir(dfs, obj, anchor_p, nr_p, dirs):
        anchor = anchor_p._obj
        if anchor.da_type == B.DAOS_ANCHOR_TYPE_EOF:
            nr_p._obj.value = 0
            return 0
        _dirent_batch(["alive", "vanished", "also_alive"], dirs)
        nr_p._obj.value = 3
        anchor.da_type = B.DAOS_ANCHOR_TYPE_EOF
        return 0

    libdfs.dfs_readdir.side_effect = fake_readdir

    def fake_stat(dfs, parent, name, stbuf_p):
        return errno.ENOENT if name == b"vanished" else 0

    libdfs.dfs_stat.side_effect = fake_stat

    with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
        entries = B.dfs_readdirplus(MagicMock(), MagicMock())

    names = [e.name for e in entries]
    assert names == ["alive", "also_alive"], "the unlinked entry should be dropped, not fatal"
    assert libdfs.dfs_readdir.called, "did not fall back to plain readdir"


def test_readdirplus_restarts_rather_than_resuming_after_failure():
    """The failed batch's list_dkey already advanced the anchor before the
    stat failed. Resuming from it would silently skip that batch's entries -
    quietly losing files is worse than the extra work of restarting.

    The advanced state must be *distinguishable* from a fresh anchor, which
    the first version of this test got wrong: it simulated "advanced" with
    ``da_type = 0``, but ``DAOS_ANCHOR_TYPE_ZERO`` is 0, so that is exactly
    what a fresh memset anchor already holds. The assertion below was then
    satisfied identically by restarting and by resuming, and a variant that
    resumed from the caller's anchor passed the whole suite. An
    ``_ANCHOR_ADVANCED`` marker makes the two outcomes different values.
    """
    libdfs = MagicMock()
    seen_anchor_types = []

    # Any non-zero, non-EOF type: what matters is that it is not what a fresh
    # anchor looks like.
    _ANCHOR_ADVANCED = 1
    assert _ANCHOR_ADVANCED not in (B.DAOS_ANCHOR_TYPE_ZERO, B.DAOS_ANCHOR_TYPE_EOF)

    def failing_readdirplus(dfs, obj, anchor_p, nr_p, dirs, stats):
        anchor_p._obj.da_type = _ANCHOR_ADVANCED  # advanced, as DAOS leaves it
        return errno.ENOENT

    libdfs.dfs_readdirplus.side_effect = failing_readdirplus

    def fake_readdir(dfs, obj, anchor_p, nr_p, dirs):
        seen_anchor_types.append(anchor_p._obj.da_type)
        nr_p._obj.value = 0
        anchor_p._obj.da_type = B.DAOS_ANCHOR_TYPE_EOF
        return 0

    libdfs.dfs_readdir.side_effect = fake_readdir

    with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
        B.dfs_readdirplus(MagicMock(), MagicMock())

    assert seen_anchor_types, "the fallback never ran"
    assert seen_anchor_types[0] != B.DAOS_ANCHOR_TYPE_EOF
    assert seen_anchor_types[0] == B.DAOS_ANCHOR_TYPE_ZERO, (
        f"fallback resumed from the advanced anchor (da_type="
        f"{seen_anchor_types[0]}) instead of restarting from a fresh one; "
        "the failed batch's entries would be silently skipped"
    )


def test_readdirplus_keeps_the_fast_path_when_nothing_vanishes():
    """The fallback costs one stat per entry, so it must not trigger normally."""
    libdfs = MagicMock()

    def ok_readdirplus(dfs, obj, anchor_p, nr_p, dirs, stats):
        _dirent_batch(["a", "b"], dirs)
        nr_p._obj.value = 2
        anchor_p._obj.da_type = B.DAOS_ANCHOR_TYPE_EOF
        return 0

    libdfs.dfs_readdirplus.side_effect = ok_readdirplus

    with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
        entries = B.dfs_readdirplus(MagicMock(), MagicMock())

    assert [e.name for e in entries] == ["a", "b"]
    libdfs.dfs_readdir.assert_not_called()
    libdfs.dfs_stat.assert_not_called()


def test_readdirplus_still_raises_on_real_errors():
    """Tolerating ENOENT must not swallow genuine failures."""
    libdfs = MagicMock()
    libdfs.dfs_readdirplus.return_value = errno.EACCES

    with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
        with pytest.raises(PermissionError):
            B.dfs_readdirplus(MagicMock(), MagicMock())

    libdfs.dfs_readdir.assert_not_called()


def test_dfs_stat_rel_returns_none_for_missing_entry():
    libdfs = MagicMock()
    libdfs.dfs_stat.return_value = errno.ENOENT

    with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
        assert B.dfs_stat_rel(MagicMock(), MagicMock(), "gone") is None


def test_dfs_stat_rel_raises_on_other_errors():
    libdfs = MagicMock()
    libdfs.dfs_stat.return_value = errno.EACCES

    with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
        with pytest.raises(PermissionError):
            B.dfs_stat_rel(MagicMock(), MagicMock(), "denied")


# ── live: the end-to-end shapes the examples actually exercise ────────────


@pytest.mark.live
def test_recursive_get_roundtrips_a_tree(dfs_fs, tmp_path):
    """What HuggingFace load_from_disk() needs, and what every example had to
    hand-roll a download_dir() walker to avoid."""
    root = "/argonne_regression_get"
    if dfs_fs.exists(root):
        dfs_fs.rm(root, recursive=True)
    dfs_fs.mkdirs(f"{root}/nested")
    dfs_fs.pipe_file(f"{root}/top.bin", b"top")
    dfs_fs.pipe_file(f"{root}/nested/inner.bin", b"inner")

    try:
        dest = tmp_path / "pulled"
        dfs_fs.get(root, str(dest), recursive=True)

        assert (dest / "top.bin").read_bytes() == b"top"
        assert (dest / "nested" / "inner.bin").read_bytes() == b"inner"
    finally:
        dfs_fs.rm(root, recursive=True)


@pytest.mark.live
def test_listing_survives_concurrent_deletes(dfs_fs):
    """Listing a directory another process is mutating must not be fatal."""
    import threading

    root = "/argonne_regression_readdir"
    if dfs_fs.exists(root):
        dfs_fs.rm(root, recursive=True)
    dfs_fs.mkdirs(root)
    names = [f"{root}/f{i:04d}" for i in range(400)]
    for n in names:
        dfs_fs.pipe_file(n, b"x")

    errors = []
    stop = threading.Event()

    def deleter():
        for n in names[200:]:
            if stop.is_set():
                return
            try:
                dfs_fs.rm(n)
            except FileNotFoundError:
                pass

    t = threading.Thread(target=deleter)
    t.start()
    try:
        for _ in range(20):
            try:
                dfs_fs.ls(root, detail=False)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                break
    finally:
        stop.set()
        t.join(timeout=30)
        try:
            dfs_fs.rm(root, recursive=True)
        except Exception:
            pass

    assert not errors, f"ls() failed against concurrent deletes: {errors}"


# ── makedirs racing another rank ──────────────────────────────────────────


def test_makedirs_survives_a_peer_creating_an_intermediate_level(mock_fs):
    """The bug behind 'give every rank its own TOP-LEVEL directory'."""
    fs, mock_B = mock_fs
    created: set[str] = set()
    raced: list[str] = []
    lookups: dict[str, int] = {}

    def lookup(dfs, parent, name, flags):
        """lookup."""
        lookups[name] = lookups.get(name, 0) + 1
        if name in created or (name == "a" and lookups[name] > 1):
            return MagicMock(name=f"obj:{name}"), _stat(0o040755)
        raise FileNotFoundError(name)

    def mkdir(dfs, parent, name, *args, **kwargs):
        if name == "a":
            raced.append(name)
            raise FileExistsError(name)  # peer won
        created.add(name)

    mock_B.dfs_lookup_rel_path.side_effect = lookup
    mock_B.dfs_mkdir.side_effect = mkdir

    fs.makedirs("/a/b/c", exist_ok=True)

    # Guard against a vacuous pass: the race must actually have happened.
    assert raced == ["a"], "the contended mkdir never fired; test proves nothing"
    # And the walk must have continued past it.
    assert created == {"b", "c"}, f"walk aborted at the raced level, created {created}"


def test_makedirs_still_raises_when_the_target_exists_and_not_exist_ok(mock_fs):
    """Tolerating the race must not break normal exist_ok=False semantics."""
    fs, mock_B = mock_fs
    mock_B.dfs_lookup_path.return_value = (MagicMock(name="obj"), _stat(0o040755))

    with pytest.raises(FileExistsError):
        fs.makedirs("/already/there", exist_ok=False)


@pytest.mark.live
def test_concurrent_makedirs_under_a_shared_parent(dfs_fs):
    """Live proof of A3: many workers creating nested paths under one shared
    parent. This is the pattern the examples had to avoid entirely by giving
    each rank its own top-level directory."""
    import concurrent.futures

    root = "/argonne_regression_mkdir"
    if dfs_fs.exists(root):
        dfs_fs.rm(root, recursive=True)

    def make(i):
        dfs_fs.makedirs(f"{root}/shared/rank_{i:03d}/deep", exist_ok=True)
        return f"{root}/shared/rank_{i:03d}/deep"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            paths = list(ex.map(make, range(32)))

        missing = [p for p in paths if not dfs_fs.exists(p)]
        assert not missing, f"makedirs reported success but these are absent: {missing[:5]}"
    finally:
        try:
            dfs_fs.rm(root, recursive=True)
        except Exception:
            pass


@pytest.mark.live
def test_h5py_accepts_our_file_object(dfs_fs):
    """h5py accepts our file object."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    root = "/argonne_regression_h5py"
    path = f"{root}/t.h5"
    if dfs_fs.exists(root):
        dfs_fs.rm(root, recursive=True)
    dfs_fs.mkdirs(root)

    try:
        import io

        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            h5.create_dataset("temps", data=np.arange(1000).reshape(100, 10))
        dfs_fs.pipe_file(path, buf.getvalue())

        with dfs_fs.open(path, "rb") as f:
            with h5py.File(f, "r") as h5:
                assert h5["temps"].shape == (100, 10)
                assert int(h5["temps"][0, 1]) == 1
    finally:
        try:
            dfs_fs.rm(root, recursive=True)
        except Exception:
            pass
