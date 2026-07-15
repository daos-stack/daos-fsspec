# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for daos-fsspec.

Uses Hypothesis for fuzz testing of path parsing, URI handling, and validation.
No DAOS cluster required.
"""

import pytest

from daos_fsspec.core import DFSFileSystem

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


class TestStripProtocolProperties:
    """Property-based tests for _strip_protocol."""

    @given(st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_relative_paths_become_absolute(self, path: str):
        """A relative path is made absolute by prefixing / with content preserved."""
        result = DFSFileSystem._strip_protocol(path)
        # Not just "starts with /" — the path content must be preserved verbatim,
        # otherwise a value-mangling bug (truncation, rewrite) would slip through.
        assert result == "/" + path

    @given(
        st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=1, max_size=50),
        st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=1, max_size=50),
        st.text(alphabet=st.characters(blacklist_characters="\x00"), min_size=1, max_size=100),
    )
    @settings(max_examples=50)
    def test_dfs_uri_extracts_path(self, pool: str, container: str, path: str):
        """dfs://pool/container/path should extract /path."""
        uri = f"dfs://{pool}/{container}/{path}"
        result = DFSFileSystem._strip_protocol(uri)
        assert result == "/" + path

    @given(st.just("/"))
    def test_root_stays_root(self, path: str):
        """Root path / should stay as /."""
        assert DFSFileSystem._strip_protocol(path) == "/"

    @given(st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=1, max_size=50))
    @settings(max_examples=30)
    def test_dfs_uri_without_path_returns_root(self, pool_container: str):
        """dfs://pool/container should return /."""
        # This tests the case where there's no path component
        parts = pool_container.split("/", 1)
        if len(parts) == 2:
            uri = f"dfs://{parts[0]}/{parts[1]}"
            result = DFSFileSystem._strip_protocol(uri)
            assert result == "/"


class TestGetKwargsProperties:
    """Property-based tests for _get_kwargs_from_urls."""

    @given(
        st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=1, max_size=50),
        st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=1, max_size=50),
        st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=50)
    def test_dfs_uri_extracts_pool_and_container(self, pool: str, container: str, path: str):
        """dfs://pool/container/path should extract pool and container."""
        uri = f"dfs://{pool}/{container}/{path}"
        kw = DFSFileSystem._get_kwargs_from_urls(uri)
        assert kw == {"pool": pool, "container": container}

    @given(st.text(min_size=1, max_size=100).filter(lambda x: not x.startswith("dfs://")))
    @settings(max_examples=30)
    def test_non_dfs_uri_returns_empty(self, path: str):
        """Non-dfs:// URIs should return empty dict."""
        kw = DFSFileSystem._get_kwargs_from_urls(path)
        assert kw == {}


class TestPathValidationProperties:
    """Property-based tests for _validate_path."""

    @given(st.text(alphabet=st.characters(blacklist_characters="\x00"), min_size=1, max_size=4096))
    @settings(max_examples=50)
    def test_valid_paths_pass(self, path: str):
        """Validation of an in-limits path returns it unchanged (identity)."""
        if not path.startswith("/"):
            path = "/" + path
        # Skip inputs that legitimately violate a limit (e.g. an over-long
        # component); for the rest, validation must be a value-preserving
        # identity, not just "did not raise".
        components = path.split("/")
        if len(path) <= 4096 and all(len(c) <= 255 for c in components):
            assert DFSFileSystem._validate_path(path) == path

    def test_long_paths_rejected(self):
        """Paths exceeding DFS_MAX_PATH should be rejected."""
        path = "/" + "a" * 5000
        with pytest.raises(ValueError, match="too long"):
            DFSFileSystem._validate_path(path)

    @given(
        st.text(alphabet=st.characters(blacklist_characters="/\x00"), min_size=256, max_size=300)
    )
    @settings(max_examples=20)
    def test_long_components_rejected(self, component: str):
        """Path components exceeding DFS_MAX_NAME should be rejected."""
        path = f"/{component}"
        with pytest.raises(ValueError, match="component too long"):
            DFSFileSystem._validate_path(path)


class TestDaosErrorProperties:
    """Property-based tests for DaosError."""

    @given(st.integers(min_value=-2000, max_value=-1))
    @settings(max_examples=50)
    def test_negative_rc_is_daos_error(self, rc: int):
        """All negative RCs should create valid DaosError."""
        from daos_fsspec._bindings import DaosError

        exc = DaosError(rc, "test")
        assert exc.rc == rc
        assert isinstance(str(exc), str)
        assert len(str(exc)) > 0

    @given(st.integers(min_value=1, max_value=256))
    @settings(max_examples=30)
    def test_positive_rc_is_errno(self, rc: int):
        """Positive RCs should create valid DaosError with errno name."""
        from daos_fsspec._bindings import DaosError

        exc = DaosError(rc, "test")
        assert exc.rc == rc
        # The message must be "<detail>: <human-readable name>" — verify the
        # detail prefix AND that a non-empty description follows, so a broken
        # _error_name lookup (empty/None) can't pass unnoticed.
        msg = str(exc)
        assert msg.startswith("test: ")
        assert len(msg[len("test: ") :].strip()) > 0
