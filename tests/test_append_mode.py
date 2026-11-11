# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for append mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings

_CHUNK = 1 << 24


class _FakeDaosError(Exception):
    is_retryable = False


@pytest.fixture
def fs_and_writes():
    """Yield (fs, writes, set_existing_size).

    ``writes`` accumulates ``(offset, length)`` for every ``dfs_write``, which
    is what these tests assert on.
    """
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.CHUNK_SIZE = _CHUNK
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755, type_name="directory", is_dir=True),
    )
    mock_B.dfs_open_file.return_value = MagicMock(name="created")

    writes: list[tuple[int, int]] = []
    mock_B.dfs_write.side_effect = lambda dfs, obj, off, data: writes.append((off, len(data)))

    def set_existing_size(size: int | None):
        """Make the target file exist with ``size`` bytes, or not exist."""
        if size is None:
            mock_B.dfs_lookup_rel_path.side_effect = FileNotFoundError("no such file")
        else:
            mock_B.dfs_lookup_rel_path.side_effect = None
            mock_B.dfs_lookup_rel_path.return_value = (
                MagicMock(name="existing"),
                SimpleNamespace(size=size, mode=0o100644, type_name="file", is_dir=False),
            )

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="p", container="c")
        yield fs, writes, set_existing_size
        fs._closed = True


class TestAppendToExistingFile:
    def test_the_first_write_lands_at_the_end(self, fs_and_writes):
        """The regression. Offset 0 here means the file's head is destroyed."""
        fs, writes, set_existing_size = fs_and_writes
        set_existing_size(1000)

        with fs.open("/existing.bin", "ab") as handle:
            handle.write(b"TAILDATA")

        assert writes, "no write reached libdfs"
        assert writes[0][0] == 1000, (
            f"append wrote at offset {writes[0][0]}, not 1000; this overwrites "
            "the start of an existing file"
        )
        assert writes[0][1] == len(b"TAILDATA")

    def test_successive_writes_advance_from_the_end(self, fs_and_writes):
        """Offsets must keep climbing, not restart at the recorded size."""
        fs, writes, set_existing_size = fs_and_writes
        set_existing_size(500)

        with fs.open("/existing.bin", "ab", block_size=8) as handle:
            handle.write(b"A" * 8)
            handle.flush()
            handle.write(b"B" * 8)

        offsets = [off for off, _ in writes]
        assert offsets[0] == 500, f"first append at {offsets[0]}, expected 500"
        assert offsets == sorted(offsets), f"offsets went backwards: {offsets}"
        assert len(set(offsets)) == len(offsets), f"an offset was reused: {offsets}"

    def test_the_total_bytes_written_match_the_payload(self, fs_and_writes):
        """Guards against an offset fix that drops or duplicates data."""
        fs, writes, set_existing_size = fs_and_writes
        set_existing_size(64)
        payload = b"x" * 100

        with fs.open("/existing.bin", "ab") as handle:
            handle.write(payload)

        assert sum(length for _, length in writes) == len(payload)

    @pytest.mark.parametrize("existing", [1, 4095, 1 << 20])
    def test_the_offset_is_the_existing_size(self, fs_and_writes, existing):
        fs, writes, set_existing_size = fs_and_writes
        set_existing_size(existing)

        with fs.open("/existing.bin", "ab") as handle:
            handle.write(b"z")

        assert writes[0][0] == existing


class TestAppendToNewFile:
    def test_a_new_file_still_starts_at_zero(self, fs_and_writes):
        """The case the old test covered, which must keep working.

        This is why the bug survived: offset 0 is correct here, so the only
        append test in the suite passed.
        """
        fs, writes, set_existing_size = fs_and_writes
        set_existing_size(None)

        with fs.open("/brand-new.bin", "ab") as handle:
            handle.write(b"first")

        assert writes[0][0] == 0, f"new-file append wrote at {writes[0][0]}, expected 0"


class TestWriteModeUnaffected:
    def test_wb_truncates_and_writes_from_zero(self, fs_and_writes):
        """The fix must not leak into "wb", which must overwrite from 0."""
        fs, writes, set_existing_size = fs_and_writes
        set_existing_size(9999)

        with fs.open("/existing.bin", "wb") as handle:
            handle.write(b"fresh")

        assert writes[0][0] == 0, (
            f'"wb" wrote at offset {writes[0][0]}; truncating writes must start at 0'
        )
