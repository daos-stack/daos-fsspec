# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for ls contents."""

from __future__ import annotations

import stat as stat_mod
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings


class _FakeDaosError(Exception):
    is_retryable = False


def _entry(name: str, **stat_kwargs):
    """A DirEntry whose stat values are all distinguishable."""
    defaults = dict(
        size=0,
        mode=stat_mod.S_IFREG | 0o644,
        uid=0,
        gid=0,
        nlink=1,
        mtime=0.0,
        atime=0.0,
        ctime=0.0,
        is_dir=False,
        type_name="file",
    )
    defaults.update(stat_kwargs)
    return SimpleNamespace(name=name, stat=SimpleNamespace(**defaults))


_ENTRIES = [
    _entry(
        "alpha.bin",
        size=1234,
        mode=stat_mod.S_IFREG | 0o600,
        uid=1001,
        gid=2001,
        mtime=1700000001.5,
        atime=1700000002.5,
        type_name="file",
    ),
    _entry(
        "subdir",
        size=4096,
        mode=stat_mod.S_IFDIR | 0o750,
        uid=1002,
        gid=2002,
        mtime=1700000003.5,
        atime=1700000004.5,
        is_dir=True,
        type_name="directory",
    ),
    _entry(
        "link",
        size=17,
        mode=stat_mod.S_IFLNK | 0o777,
        uid=1003,
        gid=2003,
        mtime=1700000005.5,
        atime=1700000006.5,
        type_name="other",
    ),
]


@pytest.fixture
def fs_with_entries():
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    # A complete stat: info() reads every field, so a partial namespace makes
    # tests fail on AttributeError rather than on the behaviour under test.
    _dir_stat = SimpleNamespace(
        size=4096,
        mode=stat_mod.S_IFDIR | 0o750,
        uid=1002,
        gid=2002,
        nlink=2,
        mtime=1700000003.5,
        atime=1700000004.5,
        ctime=1700000003.5,
        is_dir=True,
        type_name="directory",
    )

    # One handle per path, and entries only for "/d". Returning the same list
    # for every directory makes find() recurse into "subdir" forever; a
    # RecursionError that looks like a product bug and is a mock artefact.
    # Keying on the handle rather than on call order keeps repeated ls() calls
    # answering the same way, which several tests here depend on.
    _handles: dict[str, MagicMock] = {}

    def _lookup(dfs, path, flags=0, *args, **kwargs):
        handle = _handles.setdefault(path, MagicMock(name=f"obj:{path}"))
        return handle, _dir_stat

    def _readdirplus(dfs, obj, *args, **kwargs):
        return _ENTRIES if obj is _handles.get("/d") else []

    mock_B.dfs_lookup_path.side_effect = _lookup
    mock_B.dfs_readdirplus.side_effect = _readdirplus

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="p", container="c")
        yield fs, mock_B
        fs._closed = True


def _by_name(listing):
    return {e["name"]: e for e in listing}


class TestDetailedListingValues:
    def test_sizes_are_the_real_sizes(self, fs_with_entries):
        """sizes are the real sizes."""
        fs, _ = fs_with_entries

        entries = _by_name(fs.ls("/d", detail=True))

        assert entries["/d/alpha.bin"]["size"] == 1234
        assert entries["/d/subdir"]["size"] == 4096
        assert entries["/d/link"]["size"] == 17

    def test_types_distinguish_files_directories_and_symlinks(self, fs_with_entries):
        """types distinguish files directories and symlinks."""
        fs, _ = fs_with_entries

        entries = _by_name(fs.ls("/d", detail=True))

        assert entries["/d/alpha.bin"]["type"] == "file"
        assert entries["/d/subdir"]["type"] == "directory"
        assert entries["/d/link"]["type"] == "other"

    def test_modes_are_per_entry(self, fs_with_entries):
        fs, _ = fs_with_entries

        entries = _by_name(fs.ls("/d", detail=True))

        assert entries["/d/alpha.bin"]["mode"] == stat_mod.S_IFREG | 0o600
        assert entries["/d/subdir"]["mode"] == stat_mod.S_IFDIR | 0o750

    def test_ownership_and_timestamps_are_per_entry(self, fs_with_entries):
        """Guards against a field being filled from the wrong entry."""
        fs, _ = fs_with_entries

        entries = _by_name(fs.ls("/d", detail=True))

        assert (entries["/d/alpha.bin"]["uid"], entries["/d/alpha.bin"]["gid"]) == (1001, 2001)
        assert (entries["/d/subdir"]["uid"], entries["/d/subdir"]["gid"]) == (1002, 2002)
        assert entries["/d/alpha.bin"]["mtime"] == 1700000001.5
        assert entries["/d/subdir"]["atime"] == 1700000004.5

    def test_every_entry_is_returned_exactly_once(self, fs_with_entries):
        fs, _ = fs_with_entries

        listing = fs.ls("/d", detail=True)

        names = [e["name"] for e in listing]
        assert sorted(names) == ["/d/alpha.bin", "/d/link", "/d/subdir"]
        assert len(names) == len(set(names)), f"duplicate entries: {names}"


class TestNameOnlyListing:
    def test_it_returns_the_same_paths_as_the_detailed_form(self, fs_with_entries):
        fs, _ = fs_with_entries

        names = fs.ls("/d", detail=False)
        detailed = [e["name"] for e in fs.ls("/d", detail=True)]

        assert sorted(names) == sorted(detailed)

    def test_paths_are_absolute_and_not_double_slashed(self, fs_with_entries):
        """Root is the case that produces `//name` if the join is wrong."""
        fs, _ = fs_with_entries

        for path in ("/", "/d", "/d/"):
            for name in fs.ls(path, detail=False):
                assert name.startswith("/"), f"{name!r} is not absolute"
                assert not name.startswith("//"), f"{name!r} has a doubled slash"
                assert "//" not in name, f"{name!r} contains a doubled slash"


class TestInfoAgreesWithLs:
    def test_a_directory_size_is_reported_consistently(self, fs_with_entries):
        """ls() and find(detail=True) must not disagree about one entry."""
        fs, _ = fs_with_entries

        from_ls = _by_name(fs.ls("/d", detail=True))["/d/subdir"]["size"]
        found = fs.find("/d", withdirs=True, detail=True)
        from_find = found["/d/subdir"]["size"]

        assert from_ls == from_find, (
            f"ls() reports size {from_ls} for the directory and find() reports "
            f"{from_find}; same stat, two answers"
        )
