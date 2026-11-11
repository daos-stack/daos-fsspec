# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for bindings."""

import ctypes
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
        """The message must name the operation and the code."""
        with pytest.raises(FileNotFoundError) as excinfo:
            B._check(errno.ENOENT, "lookup")

        message = str(excinfo.value)
        assert "lookup" in message, f"operation missing from {message!r}"
        assert "ENOENT" in message, f"code name missing from {message!r}"


class TestDaosError:
    """DaosError exception class."""

    def test_retryable_errors_membership_is_exactly_this(self):
        """Pin the set itself, not its self-consistency."""
        import errno

        expected = {
            B.DER_HG,  # Mercury transport error
            B.DER_BUSY,
            B.DER_TIMEDOUT,
            B.DER_UNREACH,
            errno.EINTR,
            errno.EAGAIN,
            errno.ETIMEDOUT,
        }

        assert set(B.RETRYABLE_ERRORS) == expected, (
            "the retryable set changed; added "
            f"{sorted(set(B.RETRYABLE_ERRORS) - expected)}, removed "
            f"{sorted(expected - set(B.RETRYABLE_ERRORS))}"
        )

    def test_each_retryable_code_is_reported_retryable(self):
        """The mechanism, kept separate from the membership above."""
        for code in B.RETRYABLE_ERRORS:
            assert B.DaosError(code, "test").is_retryable is True

    def test_codes_that_must_not_be_retried(self):
        """Retrying these would mask a bug or loop on a permanent failure."""
        for code in (B.DER_NO_HDL, B.DER_NO_PERM, B.DER_NONEXIST, B.DER_EXIST):
            assert B.DaosError(code, "test").is_retryable is False, (
                f"{code} became retryable; an invalid handle, a permission "
                "failure or a missing/existing path is not transient"
            )

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

    def test_struct_stat_matches_the_x86_64_layout(self):
        """Size *and* offsets, unconditionally."""
        assert B.sizeof(B.struct_stat) == 144

        for field, offset in (
            ("st_dev", 0),
            ("st_ino", 8),
            ("st_nlink", 16),
            ("st_mode", 24),
            ("st_uid", 28),
            ("st_gid", 32),
            ("st_size", 48),
        ):
            actual = getattr(B.struct_stat, field).offset
            assert actual == offset, (
                f"struct_stat.{field} is at offset {actual}, expected {offset} "
                "-- the declaration drifted from glibc's x86_64 struct stat"
            )

    def test_struct_stat_st_mode_offset(self):
        """st_mode should be at offset 24."""
        mode_field = getattr(B.struct_stat, "st_mode")
        assert mode_field.offset == 24

    def test_struct_stat_st_size_offset(self):
        """st_size should be at offset 48."""
        size_field = getattr(B.struct_stat, "st_size")
        assert size_field.offset == 48


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
        # Inspect the struct_stat actually written: not just "was called". A bug
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

        # The arguments, not just the call. Swapping offset and length --
        # punching the wrong region of a file; passed assert_called_once().
        # Compared by value because the offsets arrive as c_ulong wrappers,
        # which never equal a plain int.
        mock_libdfs.dfs_punch.assert_called_once()
        args = mock_libdfs.dfs_punch.call_args.args
        assert args[0] is mock_dfs and args[1] is mock_obj
        assert args[2].value == 100, f"offset was {args[2].value}, expected 100"
        assert args[3].value == 200, f"length was {args[3].value}, expected 200"


# ---------------------------------------------------------------------------
# dfs_suggest_oclass + argtype-consistency tests (mock-based)
# ---------------------------------------------------------------------------


class TestArgtypeValueConsistency:
    """The value an impl constructs must match the argtype it declares.

    The mock-based tests use a MagicMock for libdfs, which does NOT enforce
    argtypes: so a mismatch (e.g. passing a c_uint16 value into a c_uint32
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

        BIG_OCLASS = 16777217  # 0x01000001: a real DAOS class id (e.g. SX-ish)

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

        # Ignoring the argument entirely and passing b"/WRONG" passed the old
        # assert_called_once().
        mock_libdfs.dfs_set_prefix.assert_called_once_with(mock_dfs, b"/prefix")


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

        # Swapping both the parents and the names; exchanging the wrong pair
        # of entries; passed the old assert_called_once().
        mock_libdfs.dfs_exchange.assert_called_once_with(
            mock_dfs, mock_parent1, b"a", mock_parent2, b"b"
        )


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

        from daos_fsspec._runtime import Runtime

        with patch.object(B, "_runtime", Runtime()), patch("atexit.register"):
            B.daos_init()
            B.daos_init()
            assert mock_libdaos.daos_init.call_count == 1

    @patch("daos_fsspec._bindings._ensure_libs")
    def test_daos_init_error_does_not_claim_a_reference(self, mock_ensure):
        from daos_fsspec._runtime import Runtime

        native = MagicMock()
        native.daos_init.return_value = B.DER_ALREADY
        mock_ensure.return_value = (native, MagicMock())
        with patch.object(B, "_runtime", Runtime()), patch("atexit.register"):
            with pytest.raises(B.DaosError):
                B.daos_init()
            B.daos_fini()
            native.daos_fini.assert_not_called()


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
        result = B.dfs_access(mock_dfs, mock_obj, "file.txt", os.W_OK)
        assert result is True

        # The mask has to reach libdfs. Hardcoding it to 0 would turn every
        # W_OK/X_OK question into "does this exist?" and answer yes, which
        # passed all three of these tests.
        args = mock_libdfs.dfs_access.call_args.args
        assert args[2] == b"file.txt", f"name not forwarded: {args[2]!r}"
        mask = args[3].value if hasattr(args[3], "value") else args[3]
        assert mask == os.W_OK, f"mask reached libdfs as {mask}, expected {os.W_OK}"

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

        # Write the out-param, as libdfs does. Without this the value is
        # always 0 and `assert isinstance(result, int)` held even for
        # `lambda dfs, obj: 0`.
        def fill(dfs, obj, out):
            out._obj.value = 123456
            return 0

        mock_libdfs.dfs_get_size.side_effect = fill

        result = B.dfs_get_size(MagicMock(), MagicMock())

        assert result == 123456, f"size came back as {result}, not the out-param value"

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
# ABI layout lock: struct field offsets / sizes
# ---------------------------------------------------------------------------


class TestStructABI:
    """Lock ctypes struct layouts to the DAOS 2.6.x C ABI.

    Every offset/size below was verified with a compiled ``offsetof()`` probe
    against the DAOS 2.6.4 headers (daos_types.h, daos_fs.h, gurt/types.h) on
    x86_64. These run with no DAOS installed: ctypes computes the layout purely
    in Python: so they guard the FFI contract in plain CI. A struct field that
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
        # pi_rebuild_st@136. The full 192 bytes matter: DAOS writes the whole
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
        # s_total at offset 0 of ps_space, s_free at 16: what pool_query reads.
        sub = self._layout(B._daos_space)
        assert sub["s_total"] == (0, 16)
        assert sub["s_free"] == (16, 16)


class TestSameHandle:
    """Handle comparison must be by address, not by Python wrapper identity.

    ``dfs_obj_p`` is ``c_void_p``, which defines no ``__eq__``, so ``==`` on
    two handles falls back to object identity. Every "is this the root?" check
    written as ``handle != self._root`` was therefore asking the wrong
    question. It happened to work only because ``_resolve_path`` returns
    ``self._root`` itself; a refactor that looked the root up again would have
    released the live root handle and turned the next operation into a
    use-after-free.
    """

    def test_two_wrappers_at_the_same_address_are_the_same_handle(self):
        a = B.dfs_obj_p(0x1234)
        b = B.dfs_obj_p(0x1234)

        assert a is not b, "test needs two distinct wrapper objects"
        assert a != b, "ctypes changed: == is no longer identity-based"
        assert B.same_handle(a, b) is True, "address comparison failed"

    def test_different_addresses_are_different_handles(self):
        assert B.same_handle(B.dfs_obj_p(0x1234), B.dfs_obj_p(0x5678)) is False

    def test_identity_still_counts(self):
        handle = B.dfs_obj_p(0x1234)
        assert B.same_handle(handle, handle) is True

    def test_none_is_never_the_same_as_a_handle(self):
        assert B.same_handle(None, B.dfs_obj_p(0x1234)) is False
        assert B.same_handle(B.dfs_obj_p(0x1234), None) is False

    def test_two_nulls_compare_equal(self):
        """Both NULL means neither refers to an object; treat them as equal."""
        assert B.same_handle(B.dfs_obj_p(), B.dfs_obj_p()) is True

    def test_mocks_fall_back_to_identity(self):
        """The test suite passes MagicMocks as handles; they must not collide."""
        from unittest.mock import MagicMock

        one, two = MagicMock(), MagicMock()
        assert B.same_handle(one, one) is True
        assert B.same_handle(one, two) is False


class TestObjInfoStruct:
    """dfs_obj_info_t layout, which a live value assertion used to guard badly."""

    def test_obj_info_struct_field_order(self):
        """Offsets, which is what a swap actually changes."""
        for field, offset in (
            ("doi_oclass_id", 0),
            ("doi_chunk_size", 8),
            ("doi_dir_oclass_id", 16),
            ("doi_file_oclass_id", 20),
            ("doi_oid_lo", 24),
            ("doi_oid_hi", 32),
        ):
            actual = getattr(B.dfs_obj_info_t, field).offset
            assert actual == offset, (
                f"dfs_obj_info_t.{field} is at offset {actual}, expected "
                f"{offset}; the declaration drifted from DAOS's struct"
            )

    def test_oid_low_word_precedes_the_high_word(self):
        """Stated separately because it is the counter-intuitive part.

        Reading these in the natural hi-then-lo order swaps every object ID.
        """
        assert B.dfs_obj_info_t.doi_oid_lo.offset < B.dfs_obj_info_t.doi_oid_hi.offset, (
            "doi_oid_hi now precedes doi_oid_lo; daos_obj_id_t is "
            "{ uint64 lo; uint64 hi; } and every OID would be byte-swapped"
        )

    def test_the_mapping_to_objinfo_keeps_the_words_apart(self):
        """And the struct->dataclass copy must not cross the two words."""
        from unittest.mock import MagicMock, patch

        raw = B.dfs_obj_info_t()
        raw.doi_oclass_id = 7
        raw.doi_chunk_size = 1 << 20
        raw.doi_dir_oclass_id = 0
        raw.doi_file_oclass_id = 0
        raw.doi_oid_lo = 0  # the value that broke the live assertion
        raw.doi_oid_hi = 937030204973294137

        libdfs = MagicMock()

        def fake_get_info(dfs, obj, info_ref):
            ctypes.memmove(ctypes.byref(info_ref._obj), ctypes.byref(raw), ctypes.sizeof(raw))
            return 0

        libdfs.dfs_obj_get_info.side_effect = fake_get_info

        with patch("daos_fsspec._bindings._ensure_libs", return_value=(MagicMock(), libdfs)):
            info = B.dfs_obj_get_info(MagicMock(), MagicMock())

        assert info.oid_hi == 937030204973294137, "the high word was read from the wrong field"
        assert info.oid_lo == 0, "a zero low word must survive the copy"
        assert info.oclass_id == 7
        assert info.chunk_size == 1 << 20
