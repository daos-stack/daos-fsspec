# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for daos_fsspec._bindings (ctypes FFI to libdaos.so + libdfs.so).

These tests use mocks to avoid requiring actual DAOS libraries.
"""

import errno
import os
from ctypes import sizeof as ctypes_sizeof
from unittest.mock import MagicMock, patch

import pytest

from daos_fsspec import _bindings as B

# ---------------------------------------------------------------------------
# _error_name / _check tests
# ---------------------------------------------------------------------------


class TestErrorName:
    """_error_name returns human-readable names."""

    def test_known_der_code(self):
        assert "DER_SUCCESS" in B._error_name(B.DER_SUCCESS)

    def test_known_errno(self):
        result = B._error_name(errno.ENOENT)
        assert "ENOENT" in result

    def test_unknown_positive(self):
        result = B._error_name(9999)
        assert "error" in result

    def test_unknown_negative(self):
        result = B._error_name(-9999)
        assert "error code" in result


class TestCheck:
    """_check maps DAOS/errno codes to Python exceptions."""

    def test_success(self):
        B._check(0)  # Should not raise

    def test_enoent_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            B._check(errno.ENOENT, "test")

    def test_eexist_raises_file_exists(self):
        with pytest.raises(FileExistsError):
            B._check(errno.EEXIST, "test")

    def test_eacces_raises_permission_error(self):
        with pytest.raises(PermissionError):
            B._check(errno.EACCES, "test")

    def test_enotdir_raises_not_a_directory(self):
        with pytest.raises(NotADirectoryError):
            B._check(errno.ENOTDIR, "test")

    def test_eisdir_raises_is_a_directory(self):
        with pytest.raises(IsADirectoryError):
            B._check(errno.EISDIR, "test")

    def test_enotempty_raises_oserror(self):
        with pytest.raises(OSError):
            B._check(errno.ENOTEMPTY, "test")

    def test_der_nonexist_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            B._check(B.DER_NONEXIST, "test")

    def test_der_exist_raises_file_exists(self):
        with pytest.raises(FileExistsError):
            B._check(B.DER_EXIST, "test")

    def test_der_no_perm_raises_permission_error(self):
        with pytest.raises(PermissionError):
            B._check(B.DER_NO_PERM, "test")

    def test_der_no_entry_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            B._check(B.DER_NO_ENTRY, "test")

    def test_unknown_der_raises_daos_error(self):
        with pytest.raises(B.DaosError):
            B._check(B.DER_NOSPACE, "test")

    def test_error_detail_message(self):
        try:
            B._check(errno.ENOENT, "lookup")
        except FileNotFoundError as e:
            assert "lookup" in str(e)
            assert "ENOENT" in str(e)


class TestDaosError:
    """DaosError exception class."""

    def test_retryable_errors(self):
        for code in B.RETRYABLE_ERRORS:
            exc = B.DaosError(code, "test")
            assert exc.is_retryable is True

    def test_non_retryable_error(self):
        exc = B.DaosError(B.DER_NONEXIST, "test")
        assert exc.is_retryable is False

    def test_str_representation(self):
        exc = B.DaosError(B.DER_UNREACH, "connect")
        assert "connect" in str(exc)
        assert "DER_UNREACH" in str(exc)


# ---------------------------------------------------------------------------
# _parse_stat tests
# ---------------------------------------------------------------------------


class TestParseStat:
    """_parse_stat converts struct_stat to StatResult."""

    def test_parse_stat(self):
        st = B.struct_stat()
        st.st_size = 1024
        st.st_mode = 0o100644
        st.st_uid = 1000
        st.st_gid = 1000
        st.st_nlink = 1
        st.st_mtime_sec = 1700000000
        st.st_mtime_nsec = 500_000_000
        st.st_atime_sec = 1700000000
        st.st_atime_nsec = 500_000_000
        st.st_ctime_sec = 1700000000
        st.st_ctime_nsec = 500_000_000

        result = B._parse_stat(st)
        assert result.size == 1024
        assert result.mode == 0o100644
        assert result.uid == 1000
        assert result.gid == 1000
        assert result.nlink == 1
        assert result.mtime == pytest.approx(1700000000.5)
        assert result.atime == pytest.approx(1700000000.5)
        assert result.ctime == pytest.approx(1700000000.5)

    def test_stat_result_is_dir(self):
        result = B.StatResult(
            size=0, mode=0o40755, uid=0, gid=0, nlink=2, mtime=0.0, atime=0.0, ctime=0.0
        )
        assert result.is_dir is True
        assert result.is_file is False
        assert result.is_link is False
        assert result.type_name == "directory"

    def test_stat_result_is_file(self):
        result = B.StatResult(
            size=100, mode=0o100644, uid=0, gid=0, nlink=1, mtime=0.0, atime=0.0, ctime=0.0
        )
        assert result.is_dir is False
        assert result.is_file is True
        assert result.is_link is False
        assert result.type_name == "file"

    def test_stat_result_is_link(self):
        result = B.StatResult(
            size=0, mode=0o120000, uid=0, gid=0, nlink=1, mtime=0.0, atime=0.0, ctime=0.0
        )
        assert result.is_link is True
        assert result.is_dir is False
        assert result.is_file is False
        assert result.type_name == "other"


# ---------------------------------------------------------------------------
# _anchor_is_eof tests
# ---------------------------------------------------------------------------


class TestAnchorIsEof:
    """_anchor_is_eof checks for EOF anchor type."""

    def test_eof(self):
        anchor = B.daos_anchor_t()
        anchor.da_type = B.DAOS_ANCHOR_TYPE_EOF
        assert B._anchor_is_eof(anchor) is True

    def test_not_eof(self):
        anchor = B.daos_anchor_t()
        anchor.da_type = B.DAOS_ANCHOR_TYPE_ZERO
        assert B._anchor_is_eof(anchor) is False

    def test_other_type(self):
        anchor = B.daos_anchor_t()
        anchor.da_type = 99
        assert B._anchor_is_eof(anchor) is False


# ---------------------------------------------------------------------------
# struct_stat validation tests
# ---------------------------------------------------------------------------


class TestStructStatValidation:
    """struct_stat size and field offsets."""

    def test_struct_stat_size(self):
        """struct_stat should be 144 bytes on supported architectures."""
        import platform

        arch = platform.machine()
        expected_sizes = {
            "x86_64": 144,
            "aarch64": 144,
            "ppc64le": 144,
            "s390x": 144,
        }
        if arch in expected_sizes:
            assert B.sizeof(B.struct_stat) == expected_sizes[arch]

    def test_struct_stat_st_mode_offset(self):
        """st_mode should be at offset 24."""
        mode_field = getattr(B.struct_stat, "st_mode")
        assert mode_field.offset == 24

    def test_struct_stat_st_size_offset(self):
        """st_size should be at offset 48."""
        size_field = getattr(B.struct_stat, "st_size")
        assert size_field.offset == 48


class TestStatLayoutGate:
    """The import-time ABI safety gate (_validate_stat_layout).

    It must reject a size mismatch on a known architecture (where DFS I/O would
    otherwise read garbage) and merely warn — not crash — on an unknown arch.
    """

    def test_running_arch_passes(self):
        # Exactly what runs at import: the real running arch + our struct_stat
        # size must not raise (known arch validates; unknown arch only warns).
        import platform

        B._validate_stat_layout(platform.machine(), B.sizeof(B.struct_stat))

    def test_known_arch_correct_size_ok(self):
        B._validate_stat_layout("x86_64", 144)  # matches the table -> no raise

    def test_known_arch_size_mismatch_raises(self):
        with pytest.raises(OSError, match="struct_stat size mismatch"):
            B._validate_stat_layout("x86_64", 999)

    def test_unknown_arch_warns_not_raises(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="daos_fsspec._bindings"):
            B._validate_stat_layout("sparc64", 200)  # not in the table
        assert "Unknown architecture" in caplog.text
        assert "sparc64" in caplog.text


# ---------------------------------------------------------------------------
# StatResult properties
# ---------------------------------------------------------------------------


class TestStatResultProperties:
    """StatResult property tests."""

    def test_type_name_dir(self):
        r = B.StatResult(
            size=0, mode=0o40755, uid=0, gid=0, nlink=2, mtime=0.0, atime=0.0, ctime=0.0
        )
        assert r.type_name == "directory"

    def test_type_name_file(self):
        r = B.StatResult(
            size=100, mode=0o100644, uid=0, gid=0, nlink=1, mtime=0.0, atime=0.0, ctime=0.0
        )
        assert r.type_name == "file"

    def test_type_name_link(self):
        r = B.StatResult(
            size=0, mode=0o120000, uid=0, gid=0, nlink=1, mtime=0.0, atime=0.0, ctime=0.0
        )
        assert r.type_name == "other"


# ---------------------------------------------------------------------------
# DirEntry / ContainerInfo / PoolSpace dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Data class instantiations."""

    def test_dir_entry(self):
        entry = B.DirEntry(name="test.txt", stat=None)
        assert entry.name == "test.txt"
        assert entry.stat is None

    def test_dir_entry_with_stat(self):
        stat = B.StatResult(
            size=100, mode=0o100644, uid=0, gid=0, nlink=1, mtime=0.0, atime=0.0, ctime=0.0
        )
        entry = B.DirEntry(name="test.txt", stat=stat)
        assert entry.stat.size == 100

    def test_container_info(self):
        info = B.ContainerInfo(uuid="12345678-1234-1234-1234-123456789abc", label="test-cont")
        assert info.label == "test-cont"

    def test_pool_space(self):
        space = B.PoolSpace(scm_total=1024, scm_free=512, nvme_total=2048, nvme_free=1024)
        assert space.scm_total == 1024
        assert space.nvme_total == 2048

    def test_mount_attr(self):
        attr = B.MountAttr(
            oclass_id=0, dir_oclass_id=1, file_oclass_id=2, chunk_size=1048576, mode=0
        )
        assert attr.chunk_size == 1048576

    def test_obj_info(self):
        info = B.ObjInfo(
            oclass_id=7, dir_oclass_id=0, file_oclass_id=0, chunk_size=4194304, oid_hi=1, oid_lo=2
        )
        assert info.oid_hi == 1


# ---------------------------------------------------------------------------
# Constants and flags
# ---------------------------------------------------------------------------


class TestConstants:
    """DAOS constants and flags."""

    def test_der_codes_are_negative(self):
        """DAOS error codes should be negative."""
        assert B.DER_SUCCESS == 0
        assert B.DER_NO_PERM < 0
        assert B.DER_UNREACH < 0
        assert B.DER_HG < 0

    def test_retryable_errors_is_frozenset(self):
        assert isinstance(B.RETRYABLE_ERRORS, frozenset)

    def test_chunk_size(self):
        assert B.CHUNK_SIZE == 16 * 1024 * 1024  # 16MB
        assert B._CHUNK_SIZE == B.CHUNK_SIZE

    def test_xattr_constants(self):
        assert B.XATTR_CREATE == 1
        assert B.XATTR_REPLACE == 2
        assert B.DFS_MAX_XATTR_NAME == 255
        assert B.DFS_MAX_XATTR_LEN == 65536

    def test_dfs_max_fsize(self):
        assert B.DFS_MAX_FSIZE == (1 << 64) - 1

    def test_feature_flags_are_bools(self):
        flags = [
            B._HAS_DFS_CONNECT,
            B._HAS_SNAP,
            B._HAS_LISTIO,
            B._HAS_PUNCH,
            B._HAS_SET_PREFIX,
            B._HAS_SUGGEST_OCLASS,
            B._HAS_DUP,
            B._HAS_CONT_CHECK,
            B._HAS_DIR_ANCHOR_SET,
            B._HAS_OBJ_ANCHOR_SET,
            B._HAS_ITERATE,
        ]
        for flag in flags:
            assert isinstance(flag, bool)


# ---------------------------------------------------------------------------
# dfs_setattr tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsSetattr:
    """dfs_setattr with various attribute combinations."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_setattr_mtime(self, mock_ensure):
        import ctypes

        mock_libdfs = MagicMock()
        mock_libdfs.dfs_osetattr.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        B.dfs_setattr(mock_dfs, mock_obj, mtime=1700000000.5)

        mock_libdfs.dfs_osetattr.assert_called_once()
        # Inspect the struct_stat actually written — not just "was called". A bug
        # putting the time in the wrong field or botching the sec/nsec split
        # would pass a call-happened-only assertion.
        stbuf_ref = mock_libdfs.dfs_osetattr.call_args[0][2]
        st = ctypes.cast(stbuf_ref, ctypes.POINTER(B.struct_stat)).contents
        assert st.st_mtime_sec == 1700000000
        assert st.st_mtime_nsec == 500_000_000
        flags = mock_libdfs.dfs_osetattr.call_args[0][3]
        assert flags & B.DFS_SET_ATTR_MTIME
        assert not (flags & B.DFS_SET_ATTR_SIZE)

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_setattr_no_flags(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_osetattr.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        B.dfs_setattr(mock_dfs, mock_obj)
        mock_libdfs.dfs_osetattr.assert_not_called()

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_setattr_all_flags(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_osetattr.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        B.dfs_setattr(mock_dfs, mock_obj, mtime=1700000000, atime=1700000001, file_size=100)

        mock_libdfs.dfs_osetattr.assert_called_once()
        call_args = mock_libdfs.dfs_osetattr.call_args
        flags = call_args[0][3]
        assert flags & B.DFS_SET_ATTR_MTIME
        assert flags & B.DFS_SET_ATTR_ATIME
        assert flags & B.DFS_SET_ATTR_SIZE


# ---------------------------------------------------------------------------
# dfs_xattr tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsXattr:
    """Extended attribute operations (mock-based)."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_setxattr_name_too_long(self, mock_ensure):
        mock_ensure.return_value = (MagicMock(), MagicMock())
        mock_dfs = MagicMock()
        mock_obj = MagicMock()

        with pytest.raises(ValueError, match="xattr name too long"):
            B.dfs_setxattr(mock_dfs, mock_obj, "x" * 300, b"val")

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_setxattr_value_too_long(self, mock_ensure):
        mock_ensure.return_value = (MagicMock(), MagicMock())
        mock_dfs = MagicMock()
        mock_obj = MagicMock()

        with pytest.raises(ValueError, match="xattr value too long"):
            B.dfs_setxattr(mock_dfs, mock_obj, "user.key", b"x" * 70000)

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_listxattr_empty(self, mock_ensure):
        mock_libdfs = MagicMock()
        # The size-probe call leaves sz at 0 (no xattrs) -> empty list, no second
        # call. return_value=0 with sz untouched models that.
        mock_libdfs.dfs_listxattr.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        result = B.dfs_listxattr(MagicMock(), MagicMock())
        assert result == []
        # Only the size probe ran; the buffer-read call must not happen.
        assert mock_libdfs.dfs_listxattr.call_count == 1

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_listxattr_nonempty(self, mock_ensure):
        # Exercise the real NUL-separated parse path (previously never covered):
        # probe writes the size, the read fills a NUL-separated name buffer.
        import ctypes

        names = b"user.a\x00user.b\x00user.longer.key\x00"

        def side(dfs, obj, buf, sz_ref):
            sz = ctypes.cast(sz_ref, ctypes.POINTER(B.daos_size_t)).contents
            if buf is None:
                sz.value = len(names)  # size probe
            else:
                ctypes.memmove(buf, names, len(names))  # fill the buffer
            return 0

        mock_libdfs = MagicMock()
        mock_libdfs.dfs_listxattr.side_effect = side
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        result = B.dfs_listxattr(MagicMock(), MagicMock())
        assert result == ["user.a", "user.b", "user.longer.key"]


# ---------------------------------------------------------------------------
# dfs_write chunking tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsWriteChunking:
    """dfs_write must split large writes into _CHUNK_SIZE pieces at the right
    offsets and report the full byte count — the write-side mirror of
    TestDfsReadNoTruncation."""

    def test_small_write_is_single_call(self):
        chunk = B._CHUNK_SIZE
        with (
            patch.object(B, "_ensure_libs", return_value=(MagicMock(), MagicMock())),
            patch.object(B, "_dfs_write_single") as m,
        ):
            written = B.dfs_write(MagicMock(), MagicMock(), 100, b"x" * (chunk - 1))
        # Under one chunk => exactly one underlying write, at the given offset.
        assert written == chunk - 1
        assert m.call_count == 1
        # (libdfs, dfs, obj, offset, data)
        assert m.call_args.args[3] == 100
        assert m.call_args.args[4] == b"x" * (chunk - 1)

    def test_large_write_splits_at_chunk_boundaries(self):
        chunk = B._CHUNK_SIZE
        base = 4096
        # 2 full chunks + a short tail => 3 underlying writes.
        data = b"a" * chunk + b"b" * chunk + b"c" * 500
        seen = []

        def record(libdfs, dfs, obj, offset, payload):
            seen.append((offset, len(payload)))

        with (
            patch.object(B, "_ensure_libs", return_value=(MagicMock(), MagicMock())),
            patch.object(B, "_dfs_write_single", side_effect=record),
        ):
            written = B.dfs_write(MagicMock(), MagicMock(), base, data)

        # Full length reported, and offsets advance by exactly the bytes written
        # each time (base, base+chunk, base+2*chunk) — a stale-offset bug would
        # overwrite earlier chunks and fail here.
        assert written == len(data)
        assert seen == [
            (base, chunk),
            (base + chunk, chunk),
            (base + 2 * chunk, 500),
        ]


# ---------------------------------------------------------------------------
# dfs_readdirplus batch-parse tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsReaddirplusParse:
    """dfs_readdirplus must pair each dirent name with its stat and stop at the
    EOF anchor — the parse path behind ls/find/walk/du, previously live-only."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_parses_names_and_stats(self, mock_ensure):
        import ctypes

        # (name, mode, size) — one dir, two files of different sizes.
        rows = [
            ("subdir", 0o040755, 0),
            ("alpha.txt", 0o100644, 11),
            ("beta.bin", 0o100600, 4096),
        ]

        def side(dfs, obj, anchor_ref, nr_ref, dirs_array, stats_array):
            nr = ctypes.cast(nr_ref, ctypes.POINTER(ctypes.c_uint32)).contents
            for i, (name, mode, size) in enumerate(rows):
                dirs_array[i].d_name = name.encode()
                stats_array[i].st_mode = mode
                stats_array[i].st_size = size
            nr.value = len(rows)
            return 0

        mock_libdfs = MagicMock()
        mock_libdfs.dfs_readdirplus.side_effect = side
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        # First loop check enters (not EOF); second exits (EOF) after one batch.
        with patch.object(B, "_anchor_is_eof", side_effect=[False, True]):
            entries = B.dfs_readdirplus(MagicMock(), MagicMock())

        assert [e.name for e in entries] == ["subdir", "alpha.txt", "beta.bin"]
        assert entries[0].stat.is_dir
        assert entries[1].stat.is_file
        assert entries[1].stat.size == 11
        assert entries[2].stat.size == 4096

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_empty_directory_yields_no_entries(self, mock_ensure):
        import ctypes

        def side(dfs, obj, anchor_ref, nr_ref, dirs_array, stats_array):
            ctypes.cast(nr_ref, ctypes.POINTER(ctypes.c_uint32)).contents.value = 0
            return 0

        mock_libdfs = MagicMock()
        mock_libdfs.dfs_readdirplus.side_effect = side
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        with patch.object(B, "_anchor_is_eof", side_effect=[False, True]):
            entries = B.dfs_readdirplus(MagicMock(), MagicMock())

        assert entries == []


# ---------------------------------------------------------------------------
# dfs_punch tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsPunch:
    """dfs_punch (truncate) with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_punch(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_punch.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        B.dfs_punch(mock_dfs, mock_obj, 100, 200)

        mock_libdfs.dfs_punch.assert_called_once()


# ---------------------------------------------------------------------------
# dfs_suggest_oclass + argtype-consistency tests (mock-based)
# ---------------------------------------------------------------------------


class TestArgtypeValueConsistency:
    """The value an impl constructs must match the argtype it declares.

    The mock-based tests use a MagicMock for libdfs, which does NOT enforce
    argtypes — so a mismatch (e.g. passing a c_uint16 value into a c_uint32
    oclass slot) sails through them and only blows up against a real libdfs
    (ctypes.ArgumentError). These tests patch in a real ctypes CFUNCTYPE whose
    argtypes mirror _init_signatures, so the same mismatch raises in plain CI.
    """

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_dfs_mkdir_oclass_argtype(self, mock_ensure):
        import ctypes

        # 5th arg is daos_oclass_id_t == c_uint32. If dfs_mkdir wrapped oclass in
        # c_uint16 (the bug), this raises ctypes.ArgumentError.
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        seen = {}

        @proto
        def fake_mkdir(dfs, parent, name, mode, cid):
            seen["cid"] = cid
            return 0

        libdfs = MagicMock()
        libdfs.dfs_mkdir = fake_mkdir
        mock_ensure.return_value = (MagicMock(), libdfs)

        B.dfs_mkdir(ctypes.c_void_p(1), ctypes.c_void_p(2), "d", 0o755, oclass_id=70000)
        assert seen["cid"] == 70000  # also proves no 16-bit truncation (>65535)

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_dfs_open_oclass_argtype(self, mock_ensure):
        import ctypes

        # dfs_open arg 6 is daos_oclass_id_t == c_uint32 (see _init_signatures).
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_void_p,
        )
        seen = {}

        @proto
        def fake_open(dfs, parent, name, mode, flags, cid, chunk, value, obj):
            seen["cid"] = cid
            return 0

        libdfs = MagicMock()
        libdfs.dfs_open = fake_open
        mock_ensure.return_value = (MagicMock(), libdfs)

        B.dfs_open_file(
            ctypes.c_void_p(1), ctypes.c_void_p(2), "f", 0o644, os.O_RDWR, oclass_id=70000
        )
        assert seen["cid"] == 70000


class TestDfsSuggestOclass:
    """dfs_suggest_oclass with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_suggest(self, mock_ensure):
        # Write a real oclass id LARGER than 16 bits into the out-param. The fixed
        # binding reads it through a c_uint32; a c_uint16 (the old bug) would
        # truncate 16777217 -> 1. Asserting the exact value catches that.
        import ctypes

        BIG_OCLASS = 16777217  # 0x01000001 — a real DAOS class id (e.g. SX-ish)

        def fill(dfs, hint, cid_ptr):
            ctypes.cast(cid_ptr, ctypes.POINTER(ctypes.c_uint32)).contents.value = BIG_OCLASS
            return 0

        mock_libdfs = MagicMock()
        mock_libdfs.dfs_suggest_oclass.side_effect = fill
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        result = B.dfs_suggest_oclass(MagicMock(), "file:max")
        assert result == BIG_OCLASS, f"oclass truncated: {result}"


# ---------------------------------------------------------------------------
# dfs_set_prefix tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsSetPrefix:
    """dfs_set_prefix with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_set_prefix(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_set_prefix.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        B.dfs_set_prefix(mock_dfs, "/prefix")

        mock_libdfs.dfs_set_prefix.assert_called_once()


# ---------------------------------------------------------------------------
# dfs_dup tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsDup:
    """dfs_dup with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_dup(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_dup.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        result = B.dfs_dup(mock_dfs, mock_obj)
        assert result is not None
        mock_libdfs.dfs_dup.assert_called_once()


# ---------------------------------------------------------------------------
# dfs_exchange tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsExchange:
    """dfs_exchange (atomic swap) with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_exchange(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_exchange.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_parent1 = MagicMock()
        mock_parent2 = MagicMock()
        B.dfs_exchange(mock_dfs, mock_parent1, "a", mock_parent2, "b")

        mock_libdfs.dfs_exchange.assert_called_once()


# ---------------------------------------------------------------------------
# daos_init / daos_fini tests
# ---------------------------------------------------------------------------


class TestDaosInitFini:
    """daos_init / daos_fini lifecycle."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_daos_init_idempotent(self, mock_ensure):
        mock_libdaos = MagicMock()
        mock_libdaos.daos_init.return_value = 0
        mock_ensure.return_value = (mock_libdaos, MagicMock())

        orig = B._daos_initialized
        try:
            B._daos_initialized = False
            B.daos_init()
            B.daos_init()  # Should be idempotent
            assert mock_libdaos.daos_init.call_count == 1
        finally:
            B._daos_initialized = orig

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_daos_init_already(self, mock_ensure):
        mock_libdaos = MagicMock()
        mock_libdaos.daos_init.return_value = B.DER_ALREADY
        mock_ensure.return_value = (mock_libdaos, MagicMock())

        orig = B._daos_initialized
        try:
            B._daos_initialized = False
            B.daos_init()
            assert B._daos_initialized is True
        finally:
            B._daos_initialized = orig


# ---------------------------------------------------------------------------
# dfs_access tests (mock-based)
# ---------------------------------------------------------------------------


class TestDfsAccess:
    """dfs_access with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_access_granted(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_access.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        result = B.dfs_access(mock_dfs, mock_obj, "file.txt", 0)
        assert result is True

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_access_denied(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_access.return_value = errno.EACCES
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        result = B.dfs_access(mock_dfs, mock_obj, "file.txt", 2)
        assert result is False

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_access_denied_der_no_perm(self, mock_ensure):
        # A real DAOS stack reports permission failures as DER_NO_PERM (-1001),
        # not POSIX EACCES. This branch was previously untested, so a regression
        # dropping it (raising instead of returning False) would slip through.
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_access.return_value = B.DER_NO_PERM
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        result = B.dfs_access(MagicMock(), MagicMock(), "file.txt", 2)
        assert result is False


# ---------------------------------------------------------------------------
# dfs_get_size / dfs_get_chunk_size (mock-based)
# ---------------------------------------------------------------------------


class TestDfsGetSizeChunkSize:
    """dfs_get_size and dfs_get_chunk_size with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_get_size(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_get_size.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        result = B.dfs_get_size(mock_dfs, mock_obj)
        assert isinstance(result, int)

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_get_chunk_size(self, mock_ensure):
        mock_libdfs = MagicMock()
        mock_libdfs.dfs_get_chunk_size.return_value = 0
        mock_ensure.return_value = (MagicMock(), mock_libdfs)

        mock_dfs = MagicMock()
        mock_obj = MagicMock()
        result = B.dfs_get_chunk_size(mock_dfs, mock_obj)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Pool space query (mock-based)
# ---------------------------------------------------------------------------


class TestPoolQuerySpace:
    """pool_query_space with mock."""

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_pool_query_space(self, mock_ensure):
        # Write a REAL _daos_pool_info into the out-param via side_effect, so the
        # test exercises struct parsing (offset of pi_space, SCM/NVMe indices,
        # total/free) instead of asserting fabricated zeros. The previous version
        # asserted scm_total==0 — which a MagicMock produces no matter how broken
        # the struct layout is, so it could never catch an ABI error.
        import ctypes

        def fill_info(poh, tgts, info_ptr, ranks, ev):
            info = ctypes.cast(info_ptr, ctypes.POINTER(B._daos_pool_info)).contents
            # Caller must have requested DPI_SPACE, else real DAOS fills nothing.
            assert info.pi_bits & B.DPI_SPACE
            info.pi_space.ps_space.s_total[B.DAOS_MEDIA_SCM] = 1000
            info.pi_space.ps_space.s_free[B.DAOS_MEDIA_SCM] = 400
            info.pi_space.ps_space.s_total[B.DAOS_MEDIA_NVMe] = 9000
            info.pi_space.ps_space.s_free[B.DAOS_MEDIA_NVMe] = 2500
            return 0

        mock_libdaos = MagicMock()
        mock_libdaos.daos_pool_query.side_effect = fill_info
        mock_ensure.return_value = (mock_libdaos, MagicMock())

        result = B.pool_query_space(MagicMock())

        assert isinstance(result, B.PoolSpace)
        assert result.scm_total == 1000
        assert result.scm_free == 400
        assert result.nvme_total == 9000
        assert result.nvme_free == 2500


# ---------------------------------------------------------------------------
# _load_lib tests
# ---------------------------------------------------------------------------


class TestLoadLib:
    """_load_lib function."""

    def test_load_lib_not_found(self):
        """Should raise OSError when library is not found."""
        with patch("ctypes.util.find_library", return_value=None):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(OSError, match="Cannot find"):
                    B._load_lib("nonexistent_lib_xyz")


# ---------------------------------------------------------------------------
# ABI layout lock — struct field offsets / sizes
# ---------------------------------------------------------------------------


class TestStructABI:
    """Lock ctypes struct layouts to the DAOS 2.6.x C ABI.

    Every offset/size below was verified with a compiled ``offsetof()`` probe
    against the DAOS 2.6.4 headers (daos_types.h, daos_fs.h, gurt/types.h) on
    x86_64. These run with no DAOS installed — ctypes computes the layout purely
    in Python — so they guard the FFI contract in plain CI. A struct field that
    silently shifts (e.g. an oclass typed uint16 instead of uint32) reads the
    wrong bytes from a real DAOS call; only a test like this catches it without
    a live cluster.
    """

    @staticmethod
    def _layout(struct):
        return {
            name: (getattr(struct, name).offset, getattr(struct, name).size)
            for name, *_ in struct._fields_
        }

    def test_d_iov_t(self):
        assert ctypes_sizeof(B.d_iov_t) == 24
        assert self._layout(B.d_iov_t) == {
            "iov_buf": (0, 8),
            "iov_buf_len": (8, 8),
            "iov_len": (16, 8),
        }

    def test_d_sg_list_t(self):
        assert ctypes_sizeof(B.d_sg_list_t) == 16
        assert self._layout(B.d_sg_list_t) == {
            "sg_nr": (0, 4),
            "sg_nr_out": (4, 4),
            "sg_iovs": (8, 8),
        }

    def test_daos_anchor_t(self):
        # uint16 + uint16 + uint32 + uint64 + uint8[104]
        assert ctypes_sizeof(B.daos_anchor_t) == 120
        lay = self._layout(B.daos_anchor_t)
        assert lay["da_type"] == (0, 2)
        assert lay["da_flags"] == (4, 4)
        assert lay["da_sub_anchors"] == (8, 8)
        assert lay["da_buf"] == (16, 104)

    def test_struct_stat(self):
        # glibc x86_64 struct stat
        assert ctypes_sizeof(B.struct_stat) == 144
        lay = self._layout(B.struct_stat)
        assert lay["st_size"] == (48, 8)
        assert lay["st_mtime_sec"] == (88, 8)

    def test_dfs_obj_info_t(self):
        # daos_oclass_id_t is uint32; daos_obj_id_t is {lo, hi}
        assert ctypes_sizeof(B.dfs_obj_info_t) == 40
        assert self._layout(B.dfs_obj_info_t) == {
            "doi_oclass_id": (0, 4),
            "doi_chunk_size": (8, 8),
            "doi_dir_oclass_id": (16, 4),
            "doi_file_oclass_id": (20, 4),
            "doi_oid_lo": (24, 8),
            "doi_oid_hi": (32, 8),
        }

    def test_dfs_attr_t(self):
        assert ctypes_sizeof(B.dfs_attr_t) == 176
        assert self._layout(B.dfs_attr_t) == {
            "da_id": (0, 8),
            "da_chunk_size": (8, 8),
            "da_oclass_id": (16, 4),
            "da_props": (24, 8),
            "da_mode": (32, 4),
            "da_dir_oclass_id": (36, 4),
            "da_file_oclass_id": (40, 4),
            "da_hints": (44, 128),
        }

    def test_daos_pool_info_t(self):
        # daos_pool.h: sizeof(daos_pool_info_t)==192, pi_bits@40, pi_space@48,
        # pi_rebuild_st@136. The full 192 bytes matter — DAOS writes the whole
        # struct through the pointer, so an undersized buffer corrupts the heap.
        assert ctypes_sizeof(B._daos_pool_info) == 192
        lay = self._layout(B._daos_pool_info)
        assert lay["pi_uuid"] == (0, 16)
        assert lay["pi_ntargets"] == (16, 4)
        assert lay["pi_bits"] == (40, 8)
        assert lay["pi_space"][0] == 48
        assert lay["pi_rebuild_st"] == (136, 56)

    def test_daos_pool_space(self):
        assert ctypes_sizeof(B._daos_pool_space) == 88
        lay = self._layout(B._daos_pool_space)
        assert lay["ps_space"] == (0, 32)
        assert lay["ps_ntargets"] == (80, 4)
        # s_total at offset 0 of ps_space, s_free at 16 — what pool_query reads.
        sub = self._layout(B._daos_space)
        assert sub["s_total"] == (0, 16)
        assert sub["s_free"] == (16, 16)
