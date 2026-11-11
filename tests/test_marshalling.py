# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for marshalling."""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock

import pytest

from daos_fsspec import _bindings as B


def _sgl_target(sgl_ref):
    """Extract (address, buf_len, iov_len) from a byref'd d_sg_list_t."""
    sgl = sgl_ref._obj
    assert sgl.sg_nr == 1, f"expected a single-iov list, got sg_nr={sgl.sg_nr}"
    iov = sgl.sg_iovs[0]
    return iov.iov_buf, iov.iov_buf_len, iov.iov_len


class TestReadMarshalling:
    def test_the_payload_is_returned_through_the_iov(self):
        """the payload is returned through the iov."""
        payload = bytes(range(256)) * 4  # 1024 distinctive bytes

        def fake_read(dfs, obj, sgl_ref, offset, read_size_ref, ev):
            address, buf_len, _ = _sgl_target(sgl_ref)
            assert buf_len >= len(payload), f"buffer too small: {buf_len}"
            ctypes.memmove(address, payload, len(payload))
            read_size_ref._obj.value = len(payload)
            return 0

        libdfs = MagicMock()
        libdfs.dfs_read.side_effect = fake_read

        result = B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 0, len(payload))

        assert result == payload, "the bytes libdfs wrote were not returned intact"

    def test_a_short_read_is_truncated_to_the_returned_size(self):
        """The slice must use ``read_size.value``, not the requested size.

        Slicing by the requested size would append the untouched tail of the
        buffer; NUL bytes on CPython, but still fabricated data; and the
        caller cannot tell. This single expression is the short-read contract.
        """
        requested = 4096
        actual = 100

        def fake_read(dfs, obj, sgl_ref, offset, read_size_ref, ev):
            address, _, _ = _sgl_target(sgl_ref)
            ctypes.memmove(address, b"A" * actual, actual)
            read_size_ref._obj.value = actual
            return 0

        libdfs = MagicMock()
        libdfs.dfs_read.side_effect = fake_read

        result = B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 0, requested)

        assert len(result) == actual, (
            f"returned {len(result)} bytes for a {actual}-byte read; the buffer "
            "was sliced by the requested size, fabricating the difference"
        )
        assert result == b"A" * actual

    def test_an_empty_read_returns_empty_bytes(self):
        """EOF. Must be b"", not the whole untouched buffer."""

        def fake_read(dfs, obj, sgl_ref, offset, read_size_ref, ev):
            read_size_ref._obj.value = 0
            return 0

        libdfs = MagicMock()
        libdfs.dfs_read.side_effect = fake_read

        result = B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 0, 8192)

        assert result == b"", f"EOF returned {len(result)} bytes"

    def test_the_iov_lengths_describe_the_buffer(self):
        """Both iov fields must be the requested size."""
        seen = {}

        def fake_read(dfs, obj, sgl_ref, offset, read_size_ref, ev):
            _, seen["buf_len"], seen["iov_len"] = _sgl_target(sgl_ref)
            read_size_ref._obj.value = 0
            return 0

        libdfs = MagicMock()
        libdfs.dfs_read.side_effect = fake_read

        B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 0, 2048)

        assert seen["buf_len"] == 2048, f"iov_buf_len was {seen['buf_len']}"
        assert seen["iov_len"] == 2048, f"iov_len was {seen['iov_len']}"

    def test_the_offset_is_passed_as_given(self):
        seen = {}

        def fake_read(dfs, obj, sgl_ref, offset, read_size_ref, ev):
            seen["offset"] = offset.value if hasattr(offset, "value") else offset
            read_size_ref._obj.value = 0
            return 0

        libdfs = MagicMock()
        libdfs.dfs_read.side_effect = fake_read

        B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 123456, 16)

        assert seen["offset"] == 123456

    def test_a_failure_raises_rather_than_returning_garbage(self):
        """A non-zero rc must raise, not return the untouched buffer.

        PermissionError rather than DaosError: _check maps the POSIX-shaped
        DAOS codes onto their OSError subclasses, which is the mapping callers
        actually see.
        """
        libdfs = MagicMock()
        libdfs.dfs_read.return_value = B.DER_NO_PERM

        with pytest.raises(PermissionError):
            B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 0, 16)

    def test_a_transport_failure_raises_a_daos_error(self):
        """A code with no POSIX equivalent surfaces as DaosError."""
        libdfs = MagicMock()
        libdfs.dfs_read.return_value = B.DER_HG

        with pytest.raises(B.DaosError):
            B._dfs_read_single(libdfs, MagicMock(), MagicMock(), 0, 16)


class TestWriteMarshalling:
    def test_the_bytes_handed_to_libdfs_are_the_payload(self):
        payload = bytes(range(200))
        seen = {}

        def fake_write(dfs, obj, sgl_ref, offset, ev):
            address, buf_len, iov_len = _sgl_target(sgl_ref)
            seen["bytes"] = ctypes.string_at(address, iov_len)
            seen["buf_len"] = buf_len
            return 0

        libdfs = MagicMock()
        libdfs.dfs_write.side_effect = fake_write

        B._dfs_write_single(libdfs, MagicMock(), MagicMock(), 0, payload)

        assert seen["bytes"] == payload, "libdfs was handed different bytes"
        assert seen["buf_len"] == len(payload)

    def test_embedded_nuls_survive(self):
        """Binary payloads must round-trip, NULs and all."""
        payload = b"head\x00\x00middle\x00tail"
        seen = {}

        def fake_write(dfs, obj, sgl_ref, offset, ev):
            address, _, iov_len = _sgl_target(sgl_ref)
            seen["bytes"] = ctypes.string_at(address, iov_len)
            return 0

        libdfs = MagicMock()
        libdfs.dfs_write.side_effect = fake_write

        B._dfs_write_single(libdfs, MagicMock(), MagicMock(), 0, payload)

        assert seen["bytes"] == payload, (
            f"payload truncated at a NUL: {seen['bytes']!r} != {payload!r}"
        )

    def test_the_offset_is_passed_as_given(self):
        seen = {}

        def fake_write(dfs, obj, sgl_ref, offset, ev):
            seen["offset"] = offset.value if hasattr(offset, "value") else offset
            return 0

        libdfs = MagicMock()
        libdfs.dfs_write.side_effect = fake_write

        B._dfs_write_single(libdfs, MagicMock(), MagicMock(), 999, b"x")

        assert seen["offset"] == 999

    def test_a_failure_raises(self):
        libdfs = MagicMock()
        libdfs.dfs_write.return_value = B.DER_NOSPACE

        # DaosError, not OSError: DER_NOSPACE has no POSIX mapping in _check,
        # and DaosError does not subclass OSError.
        with pytest.raises(B.DaosError):
            B._dfs_write_single(libdfs, MagicMock(), MagicMock(), 0, b"x")
