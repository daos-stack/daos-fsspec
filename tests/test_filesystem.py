# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for filesystem."""

import concurrent.futures
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import patch_bindings

from daos_fsspec import _bindings as B
from daos_fsspec.async_core import DFSAsyncFileSystem

# ── Unit tests (no DAOS needed) ──────────────────────────────────────────


class TestURIParsing:
    """URI scheme parsing."""

    def test_strip_protocol(self):
        from daos_fsspec.core import DFSFileSystem

        assert DFSFileSystem._strip_protocol("dfs://pool/cont/path/to/file") == "/path/to/file"
        assert DFSFileSystem._strip_protocol("dfs://pool/cont/") == "/"
        assert DFSFileSystem._strip_protocol("dfs://pool/cont") == "/"
        assert DFSFileSystem._strip_protocol("/path/to/file") == "/path/to/file"
        assert DFSFileSystem._strip_protocol("relative") == "/relative"

    def test_get_kwargs_from_urls(self):
        from daos_fsspec.core import DFSFileSystem

        kw = DFSFileSystem._get_kwargs_from_urls("dfs://mypool/mycont/data.parquet")
        assert kw == {"pool": "mypool", "container": "mycont"}

    def test_strip_protocol_deep_path(self):
        from daos_fsspec.core import DFSFileSystem

        assert DFSFileSystem._strip_protocol("dfs://p/c/a/b/c/d") == "/a/b/c/d"

    def test_strip_protocol_root_only(self):
        from daos_fsspec.core import DFSFileSystem

        assert DFSFileSystem._strip_protocol("/") == "/"

    def test_get_kwargs_from_urls_no_container(self):
        from daos_fsspec.core import DFSFileSystem

        # dfs://mypool has pool but no container
        kw = DFSFileSystem._get_kwargs_from_urls("dfs://mypool")
        assert kw == {"pool": "mypool"}

    def test_get_kwargs_from_urls_empty(self):
        from daos_fsspec.core import DFSFileSystem

        kw = DFSFileSystem._get_kwargs_from_urls("dfs://")
        assert kw == {}

    def test_get_kwargs_from_urls_trailing_slash(self):
        from daos_fsspec.core import DFSFileSystem

        # dfs://pool/ has pool but empty container: should omit container
        kw = DFSFileSystem._get_kwargs_from_urls("dfs://mypool/")
        assert kw == {"pool": "mypool"}

    def test_get_kwargs_from_urls_no_scheme(self):
        from daos_fsspec.core import DFSFileSystem

        kw = DFSFileSystem._get_kwargs_from_urls("/local/path")
        assert kw == {}

    def test_strip_protocol_dfs_with_nul_raises(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError, match="NUL"):
            DFSFileSystem._strip_protocol("dfs://pool/cont/path\x00evil")


class TestPathValidation:
    """Path validation (no DAOS needed)."""

    def test_validate_path_normal(self):
        from daos_fsspec.core import DFSFileSystem

        assert DFSFileSystem._validate_path("/foo/bar") == "/foo/bar"
        assert DFSFileSystem._validate_path("/") == "/"

    def test_validate_path_nul_bytes(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError, match="NUL"):
            DFSFileSystem._validate_path("/foo\x00bar")

    def test_validate_path_too_long(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError, match="too long"):
            DFSFileSystem._validate_path("/" + "a" * 5000)

    def test_validate_path_component_too_long(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError, match="component too long"):
            DFSFileSystem._validate_path("/" + "a" * 300)

    def test_validate_path_empty_component_ok(self):
        from daos_fsspec.core import DFSFileSystem

        # Double slashes are normalized elsewhere
        assert DFSFileSystem._validate_path("//foo") == "//foo"


class TestDaosErrorRetryable:
    """DaosError.is_retryable property."""

    def test_retryable_error(self):
        from daos_fsspec._bindings import DER_UNREACH, DaosError

        exc = DaosError(DER_UNREACH, "test")
        assert exc.is_retryable is True

    def test_non_retryable_error(self):
        from daos_fsspec._bindings import DER_NONEXIST, DaosError

        exc = DaosError(DER_NONEXIST, "test")
        assert exc.is_retryable is False


# ── Live tests ───────────────────────────────────────────────────────────


@pytest.mark.live
class TestBasicOps:
    """Basic filesystem operations: ls, mkdir, stat, rm."""

    def test_ls_root(self, dfs_fs):
        entries = dfs_fs.ls("/")
        assert isinstance(entries, list)

    def test_mkdir_and_ls(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/subdir")
        entries = dfs_fs.ls(d)
        names = [os.path.basename(e) for e in entries]
        assert "subdir" in names

    def test_mkdir_nested(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/a/b/c", create_parents=True)
        assert dfs_fs.exists(f"{d}/a/b/c")

    def test_info(self, dfs_fs):
        d = dfs_fs._test_dir
        info = dfs_fs.info(d)
        assert info["type"] == "directory"

    def test_rmdir(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/to-delete")
        assert dfs_fs.exists(f"{d}/to-delete")
        dfs_fs.rmdir(f"{d}/to-delete")
        assert not dfs_fs.exists(f"{d}/to-delete")


@pytest.mark.live
class TestFileIO:
    """File read/write operations."""

    def test_write_and_read(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/hello.txt"
        data = b"Hello, DAOS DFS!\n"

        with dfs_fs.open(path, "wb") as f:
            f.write(data)

        with dfs_fs.open(path, "rb") as f:
            result = f.read()

        assert result == data

    def test_large_file(self, dfs_fs):
        """Write and read back a 4MB file."""
        d = dfs_fs._test_dir
        path = f"{d}/large.bin"
        data = os.urandom(4 * 1024 * 1024)

        with dfs_fs.open(path, "wb") as f:
            f.write(data)

        with dfs_fs.open(path, "rb") as f:
            result = f.read()

        assert result == data

    def test_cat(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/cat-test.txt"
        data = b"cat test data"

        with dfs_fs.open(path, "wb") as f:
            f.write(data)

        result = dfs_fs.cat(path)
        assert result == data

    def test_file_info(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/info-test.txt"
        data = b"x" * 1234

        with dfs_fs.open(path, "wb") as f:
            f.write(data)

        info = dfs_fs.info(path)
        assert info["type"] == "file"
        assert info["size"] == 1234

    def test_rm_file(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/to-delete.txt"

        with dfs_fs.open(path, "wb") as f:
            f.write(b"delete me")

        assert dfs_fs.exists(path)
        dfs_fs.rm(path)
        assert not dfs_fs.exists(path)


@pytest.mark.live
class TestMoveRename:
    """Rename/move operations."""

    def test_mv(self, dfs_fs):
        d = dfs_fs._test_dir
        src = f"{d}/src.txt"
        dst = f"{d}/dst.txt"

        with dfs_fs.open(src, "wb") as f:
            f.write(b"move me")

        dfs_fs.mv(src, dst)
        assert not dfs_fs.exists(src)
        assert dfs_fs.exists(dst)

        result = dfs_fs.cat(dst)
        assert result == b"move me"


@pytest.mark.live
class TestRecursive:
    """Recursive rm and nested directory ops."""

    def test_rm_recursive(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/tree/sub1/sub2", create_parents=True)
        with dfs_fs.open(f"{d}/tree/sub1/file.txt", "wb") as f:
            f.write(b"nested")

        dfs_fs.rm(f"{d}/tree", recursive=True)
        assert not dfs_fs.exists(f"{d}/tree")


@pytest.mark.live
class TestLsDetail:
    """ls with detail=True."""

    def test_ls_detail(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/detail-dir")
        with dfs_fs.open(f"{d}/detail-file.txt", "wb") as f:
            f.write(b"detail")

        entries = dfs_fs.ls(d, detail=True)
        names = {e["name"].split("/")[-1] for e in entries}
        assert "detail-dir" in names
        assert "detail-file.txt" in names

        dir_entry = next(e for e in entries if e["name"].endswith("detail-dir"))
        assert dir_entry["type"] == "directory"

        file_entry = next(e for e in entries if e["name"].endswith("detail-file.txt"))
        assert file_entry["type"] == "file"
        assert file_entry["size"] == 6


@pytest.mark.live
class TestThreadSafety:
    """Concurrent reads from multiple threads on the same filesystem."""

    def test_concurrent_reads(self, dfs_fs):
        d = dfs_fs._test_dir
        # Write test files
        for i in range(4):
            with dfs_fs.open(f"{d}/thread-{i}.txt", "wb") as f:
                f.write(f"thread data {i}".encode())

        def read_file(idx):
            data = dfs_fs.cat(f"{d}/thread-{idx}.txt")
            return data

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(read_file, i) for i in range(4)]
            results = [f.result() for f in futures]

        for i, result in enumerate(results):
            assert result == f"thread data {i}".encode()


# ── Context manager ──────────────────────────────────────────────────────


@pytest.mark.live
class TestContextManager:
    """Context manager support."""

    def test_context_manager(self, daos_pool, daos_container):
        from daos_fsspec.core import DFSFileSystem

        with DFSFileSystem(pool=daos_pool, container=daos_container) as fs:
            entries = fs.ls("/")
            assert isinstance(entries, list)
        # After exit, should be closed
        assert fs._closed


# ── Touch / pipe / modified / created ────────────────────────────────────


@pytest.mark.live
class TestTouchPipe:
    """touch() and pipe_file() operations."""

    def test_touch_create(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/touched.txt"
        dfs_fs.touch(path)
        assert dfs_fs.exists(path)
        info = dfs_fs.info(path)
        assert info["size"] == 0

    def test_touch_no_truncate(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/touch-notrunc.txt"
        with dfs_fs.open(path, "wb") as f:
            f.write(b"keep me")

        dfs_fs.touch(path, truncate=False)
        result = dfs_fs.cat(path)
        assert result == b"keep me"

    def test_touch_truncate(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/touch-trunc.txt"
        with dfs_fs.open(path, "wb") as f:
            f.write(b"truncate me")

        dfs_fs.touch(path, truncate=True)
        info = dfs_fs.info(path)
        assert info["size"] == 0

    def test_pipe_file(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/piped.txt"
        data = b"piped data goes here"
        dfs_fs.pipe_file(path, data)
        result = dfs_fs.cat(path)
        assert result == data

    def test_pipe_file_overwrite(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/pipe-overwrite.txt"
        dfs_fs.pipe_file(path, b"first version")
        dfs_fs.pipe_file(path, b"second version")
        result = dfs_fs.cat(path)
        assert result == b"second version"

    def test_modified(self, dfs_fs):
        import time

        d = dfs_fs._test_dir
        path = f"{d}/modified-test.txt"
        t0 = time.time()
        dfs_fs.pipe_file(path, b"data")
        mtime = dfs_fs.modified(path)
        assert isinstance(mtime, float)
        # `> 0` passes on any garbage float: assert the timestamp is real
        # wall-clock near now, which catches a wrong stat offset or sec/nsec mixup.
        assert t0 - 5 < mtime < t0 + 30

    def test_created(self, dfs_fs):
        import time

        d = dfs_fs._test_dir
        path = f"{d}/created-test.txt"
        t0 = time.time()
        dfs_fs.pipe_file(path, b"data")
        ctime = dfs_fs.created(path)
        assert isinstance(ctime, float)
        assert t0 - 5 < ctime < t0 + 30


# ── Symlinks ─────────────────────────────────────────────────────────────


@pytest.mark.live
class TestSymlinks:
    """Symlink create, read, detect."""

    def test_symlink_create_and_read(self, dfs_fs):
        d = dfs_fs._test_dir
        target = f"{d}/symlink-target.txt"
        link = f"{d}/symlink-link"

        dfs_fs.pipe_file(target, b"symlink target data")
        dfs_fs.symlink(link, target)

        assert dfs_fs.lexists(link)
        read_target = dfs_fs.readlink(link)
        assert read_target == target

    def test_islink(self, dfs_fs):
        d = dfs_fs._test_dir
        target = f"{d}/islink-target.txt"
        link = f"{d}/islink-link"

        dfs_fs.pipe_file(target, b"data")
        dfs_fs.symlink(link, target)

        assert dfs_fs.islink(link)
        assert not dfs_fs.islink(target)

    def test_rm_symlink(self, dfs_fs):
        d = dfs_fs._test_dir
        target = f"{d}/rm-sym-target.txt"
        link = f"{d}/rm-sym-link"

        dfs_fs.pipe_file(target, b"data")
        dfs_fs.symlink(link, target)
        assert dfs_fs.lexists(link)

        dfs_fs.rm(link)
        assert not dfs_fs.lexists(link)
        # Target should still exist
        assert dfs_fs.exists(target)


# ── Walk ─────────────────────────────────────────────────────────────────


@pytest.mark.live
class TestWalk:
    """walk() directory tree traversal."""

    def test_walk_basic(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/walk-tree/sub1", create_parents=True)
        dfs_fs.mkdir(f"{d}/walk-tree/sub2", create_parents=True)
        dfs_fs.pipe_file(f"{d}/walk-tree/file1.txt", b"data1")
        dfs_fs.pipe_file(f"{d}/walk-tree/sub1/file2.txt", b"data2")

        results = list(dfs_fs.walk(f"{d}/walk-tree"))
        # Should get 3 entries: walk-tree, sub1, sub2
        paths = [r[0] for r in results]
        assert f"{d}/walk-tree" in paths
        assert f"{d}/walk-tree/sub1" in paths
        assert f"{d}/walk-tree/sub2" in paths

        # Root should have file1.txt and two subdirs
        root_entry = next(r for r in results if r[0] == f"{d}/walk-tree")
        assert "file1.txt" in root_entry[2]
        assert "sub1" in root_entry[1]
        assert "sub2" in root_entry[1]

    def test_walk_maxdepth(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/walk-depth/a/b", create_parents=True)
        dfs_fs.pipe_file(f"{d}/walk-depth/a/b/deep.txt", b"deep")

        results = list(dfs_fs.walk(f"{d}/walk-depth", maxdepth=1))
        paths = [r[0] for r in results]
        assert f"{d}/walk-depth" in paths
        # Should NOT descend into a/b
        assert f"{d}/walk-depth/a/b" not in paths


# ── Error handling ───────────────────────────────────────────────────────


@pytest.mark.live
class TestErrorHandling:
    """Proper exception types for error conditions."""

    def test_not_found_raises_file_not_found(self, dfs_fs):
        with pytest.raises(FileNotFoundError):
            dfs_fs.info("/nonexistent-path-12345")

    def test_exists_returns_false_for_missing(self, dfs_fs):
        assert not dfs_fs.exists("/nonexistent-path-12345")

    def test_rmdir_nonempty(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/nonempty-dir")
        dfs_fs.pipe_file(f"{d}/nonempty-dir/file.txt", b"data")
        # Removing non-empty dir without recursive should fail
        # DAOS returns ENOTEMPTY for non-empty directories
        with pytest.raises(OSError):
            dfs_fs.rmdir(f"{d}/nonempty-dir")

    def test_read_nonexistent_file(self, dfs_fs):
        with pytest.raises(FileNotFoundError):
            dfs_fs.cat("/nonexistent-file-99999.txt")


# ── Chunked I/O (large files) ────────────────────────────────────────────


@pytest.mark.live
class TestChunkedIO:
    """Large file I/O that exercises chunked read/write paths."""

    def test_32mb_file(self, dfs_fs):
        """Write and read 32MB: crosses the 16MB chunk boundary."""
        d = dfs_fs._test_dir
        path = f"{d}/chunked-32mb.bin"
        data = os.urandom(32 * 1024 * 1024)

        with dfs_fs.open(path, "wb") as f:
            f.write(data)

        with dfs_fs.open(path, "rb") as f:
            result = f.read()

        assert len(result) == len(data)
        assert result == data

    def test_pipe_large(self, dfs_fs):
        """pipe_file with data > 16MB."""
        d = dfs_fs._test_dir
        path = f"{d}/pipe-large.bin"
        data = os.urandom(20 * 1024 * 1024)

        dfs_fs.pipe_file(path, data)
        result = dfs_fs.cat(path)

        assert len(result) == len(data)
        assert result == data


# ── Stress tests ─────────────────────────────────────────────────────────


@pytest.mark.live
class TestStress:
    """Heavy-duty tests: concurrent writers, many files, deep trees, weird names."""

    def test_many_small_files(self, dfs_fs):
        """Create and read 100 small files."""
        d = dfs_fs._test_dir
        n = 100
        dfs_fs.mkdir(f"{d}/many-files")
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/many-files/f-{i:04d}.txt", f"data-{i}".encode())

        entries = dfs_fs.ls(f"{d}/many-files")
        assert len(entries) == n

        # Read them all back
        for i in range(n):
            result = dfs_fs.cat(f"{d}/many-files/f-{i:04d}.txt")
            assert result == f"data-{i}".encode()

    def test_deep_directory_tree(self, dfs_fs):
        """Create a 20-level deep directory tree."""
        d = dfs_fs._test_dir
        deep = "/".join([f"d{i}" for i in range(20)])
        path = f"{d}/deep-tree/{deep}"
        dfs_fs.mkdir(path, create_parents=True)
        assert dfs_fs.exists(path)

        # Write a file at the bottom
        dfs_fs.pipe_file(f"{path}/bottom.txt", b"bottom")
        result = dfs_fs.cat(f"{path}/bottom.txt")
        assert result == b"bottom"

    def test_concurrent_writers(self, dfs_fs):
        """4 threads writing different files simultaneously."""
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/concurrent-write")

        def write_file(idx):
            data = os.urandom(64 * 1024)  # 64KB each
            path = f"{d}/concurrent-write/cw-{idx}.bin"
            dfs_fs.pipe_file(path, data)
            return path, data

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(write_file, i) for i in range(4)]
            results = [f.result() for f in futures]

        # Verify all files
        for path, expected in results:
            actual = dfs_fs.cat(path)
            assert actual == expected

    def test_empty_file(self, dfs_fs):
        """Create and read an empty file."""
        d = dfs_fs._test_dir
        path = f"{d}/empty.bin"
        dfs_fs.pipe_file(path, b"")
        info = dfs_fs.info(path)
        assert info["size"] == 0
        result = dfs_fs.cat(path)
        assert result == b""

    def test_binary_patterns(self, dfs_fs):
        """Write files with specific binary patterns."""
        d = dfs_fs._test_dir
        patterns = [
            ("all-zeros", b"\x00" * 4096),
            ("all-ones", b"\xff" * 4096),
            ("alternating", b"\xaa\x55" * 2048),
            ("ascending", bytes(range(256)) * 16),
        ]
        for name, data in patterns:
            path = f"{d}/pattern-{name}.bin"
            dfs_fs.pipe_file(path, data)
            result = dfs_fs.cat(path)
            assert result == data, f"Pattern {name} mismatch"

    def test_special_chars_in_names(self, dfs_fs):
        """Files with spaces, dots, dashes, underscores."""
        d = dfs_fs._test_dir
        names = [
            "file with spaces.txt",
            "file.multiple.dots.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "MiXeD.CaSe.TXT",
            "123numeric456.txt",
        ]
        for name in names:
            path = f"{d}/{name}"
            data = f"content of {name}".encode()
            dfs_fs.pipe_file(path, data)
            result = dfs_fs.cat(path)
            assert result == data, f"File {name} content mismatch"

    def test_rapid_create_delete(self, dfs_fs):
        """Rapidly create and delete the same file 20 times."""
        d = dfs_fs._test_dir
        path = f"{d}/rapid-cd.txt"
        for i in range(20):
            dfs_fs.pipe_file(path, f"iteration {i}".encode())
            assert dfs_fs.exists(path)
            dfs_fs.rm(path)
            assert not dfs_fs.exists(path)

    def test_overwrite_file_different_sizes(self, dfs_fs):
        """Overwrite a file with different sizes to test truncation."""
        d = dfs_fs._test_dir
        path = f"{d}/overwrite.bin"

        # Write large, then small
        large_data = os.urandom(100_000)
        dfs_fs.pipe_file(path, large_data)
        assert dfs_fs.cat(path) == large_data

        small_data = b"tiny"
        dfs_fs.pipe_file(path, small_data)
        result = dfs_fs.cat(path)
        assert result == small_data
        assert len(result) == 4  # Not 100000

    def test_many_directories(self, dfs_fs):
        """Create 50 sibling directories."""
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/many-dirs")
        for i in range(50):
            dfs_fs.mkdir(f"{d}/many-dirs/dir-{i:03d}")

        entries = dfs_fs.ls(f"{d}/many-dirs")
        assert len(entries) == 50

    def test_ls_detail_large(self, dfs_fs):
        """ls(detail=True) on a directory with 50 entries uses readdirplus efficiently."""
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/ls-detail-50")
        for i in range(50):
            dfs_fs.pipe_file(f"{d}/ls-detail-50/f-{i:03d}.txt", f"data-{i}".encode())

        entries = dfs_fs.ls(f"{d}/ls-detail-50", detail=True)
        assert len(entries) == 50
        # Every entry should have size > 0
        for e in entries:
            assert e["type"] == "file"
            assert e["size"] > 0

    def test_mv_across_directories(self, dfs_fs):
        """Move a file between different directories."""
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/mv-src-dir")
        dfs_fs.mkdir(f"{d}/mv-dst-dir")
        dfs_fs.pipe_file(f"{d}/mv-src-dir/file.txt", b"move across dirs")

        dfs_fs.mv(f"{d}/mv-src-dir/file.txt", f"{d}/mv-dst-dir/file.txt")
        assert not dfs_fs.exists(f"{d}/mv-src-dir/file.txt")
        assert dfs_fs.cat(f"{d}/mv-dst-dir/file.txt") == b"move across dirs"

    def test_makedirs_exist_ok(self, dfs_fs):
        """makedirs with exist_ok=True on existing path."""
        d = dfs_fs._test_dir
        path = f"{d}/makedirs-test/a/b"
        dfs_fs.makedirs(path, exist_ok=True)
        assert dfs_fs.exists(path)
        # Should not raise
        dfs_fs.makedirs(path, exist_ok=True)
        assert dfs_fs.exists(path)

    def test_rm_list_of_paths(self, dfs_fs):
        """rm() with a list of paths."""
        d = dfs_fs._test_dir
        paths = [f"{d}/rm-list-{i}.txt" for i in range(5)]
        for p in paths:
            dfs_fs.pipe_file(p, b"rm me")

        dfs_fs.rm(paths)
        for p in paths:
            assert not dfs_fs.exists(p)


# ── Live tests for optimized methods ────────────────────────────────────


@pytest.mark.live
class TestCatFileLive:
    """cat_file optimized path against real DAOS."""

    def test_cat_file_full(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/catfile-full.bin"
        data = os.urandom(1024)
        dfs_fs.pipe_file(path, data)
        result = dfs_fs.cat_file(path)
        assert result == data

    def test_cat_file_range(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/catfile-range.bin"
        data = b"ABCDEFGHIJ" * 100  # 1000 bytes
        dfs_fs.pipe_file(path, data)
        result = dfs_fs.cat_file(path, start=100, end=200)
        assert result == data[100:200]

    def test_cat_file_negative_offset(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/catfile-neg.bin"
        data = b"X" * 500
        dfs_fs.pipe_file(path, data)
        result = dfs_fs.cat_file(path, start=-10)
        assert result == data[-10:]

    def test_cat_file_empty_range(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/catfile-empty.bin"
        dfs_fs.pipe_file(path, b"data")
        result = dfs_fs.cat_file(path, start=5, end=5)
        assert result == b""

    def test_cat_file_not_found(self, dfs_fs):
        with pytest.raises(FileNotFoundError):
            dfs_fs.cat_file(f"{dfs_fs._test_dir}/nonexistent.txt")

    def test_cat_file_large_32mb(self, dfs_fs):
        """cat_file on 32MB file: exercises chunked dfs_read."""
        d = dfs_fs._test_dir
        path = f"{d}/catfile-32mb.bin"
        data = os.urandom(32 * 1024 * 1024)
        dfs_fs.pipe_file(path, data)
        result = dfs_fs.cat_file(path)
        assert len(result) == len(data)
        assert result == data

    def test_cat_file_consistency_with_cat(self, dfs_fs):
        """cat_file and cat() should return the same data."""
        d = dfs_fs._test_dir
        path = f"{d}/catfile-consistency.bin"
        data = os.urandom(4096)
        dfs_fs.pipe_file(path, data)
        assert dfs_fs.cat_file(path) == dfs_fs.cat(path)


@pytest.mark.live
class TestCatRangesLive:
    """cat_ranges against real DAOS."""

    def test_cat_ranges_basic(self, dfs_fs):
        d = dfs_fs._test_dir
        paths = []
        expected = []
        for i in range(4):
            p = f"{d}/ranges-{i}.bin"
            data = f"file-{i}-data".encode()
            dfs_fs.pipe_file(p, data)
            paths.append(p)
            expected.append(data)

        result = dfs_fs.cat_ranges(paths, [0] * 4, [len(e) for e in expected])
        assert result == expected

    def test_cat_ranges_with_offsets(self, dfs_fs):
        d = dfs_fs._test_dir
        p = f"{d}/ranges-offset.bin"
        data = b"0123456789" * 10  # 100 bytes
        dfs_fs.pipe_file(p, data)

        result = dfs_fs.cat_ranges([p, p], [0, 50], [10, 60])
        assert result[0] == data[0:10]
        assert result[1] == data[50:60]

    def test_cat_ranges_single_path(self, dfs_fs):
        d = dfs_fs._test_dir
        p = f"{d}/ranges-single.bin"
        data = b"single"
        dfs_fs.pipe_file(p, data)
        result = dfs_fs.cat_ranges([p], [0], [6])
        assert result == [data]

    def test_cat_ranges_on_error_return(self, dfs_fs):
        d = dfs_fs._test_dir
        good = f"{d}/ranges-good.bin"
        dfs_fs.pipe_file(good, b"ok")
        bad = f"{d}/ranges-missing.bin"

        result = dfs_fs.cat_ranges([good, bad], [0, 0], [2, 10], on_error="return")
        assert result[0] == b"ok"
        assert isinstance(result[1], Exception)

    def test_cat_ranges_on_error_raise(self, dfs_fs):
        d = dfs_fs._test_dir
        good = f"{d}/ranges-good2.bin"
        dfs_fs.pipe_file(good, b"ok")
        bad = f"{d}/ranges-missing2.bin"

        with pytest.raises(FileNotFoundError):
            dfs_fs.cat_ranges([good, bad], [0, 0], [2, 10], on_error="raise")

    def test_cat_ranges_preserves_order(self, dfs_fs):
        """Results are in the same order as input paths."""
        d = dfs_fs._test_dir
        paths = []
        for i in range(8):
            p = f"{d}/ranges-order-{i}.bin"
            dfs_fs.pipe_file(p, f"idx-{i}".encode())
            paths.append(p)

        result = dfs_fs.cat_ranges(paths, [0] * 8, [5] * 8)
        for i, r in enumerate(result):
            assert r == f"idx-{i}".encode()


@pytest.mark.live
class TestFindLive:
    """find() readdirplus-batched listing against real DAOS."""

    def test_find_flat(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/find-flat")
        for i in range(5):
            dfs_fs.pipe_file(f"{d}/find-flat/f-{i}.txt", f"d{i}".encode())

        result = dfs_fs.find(f"{d}/find-flat")
        assert len(result) == 5
        assert all("/find-flat/" in p for p in result)
        assert result == sorted(result)

    def test_find_nested(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/find-nest/sub1/sub2", create_parents=True)
        dfs_fs.pipe_file(f"{d}/find-nest/top.txt", b"top")
        dfs_fs.pipe_file(f"{d}/find-nest/sub1/mid.txt", b"mid")
        dfs_fs.pipe_file(f"{d}/find-nest/sub1/sub2/deep.txt", b"deep")

        result = dfs_fs.find(f"{d}/find-nest")
        assert len(result) == 3
        names = [p.split("/")[-1] for p in result]
        assert "top.txt" in names
        assert "mid.txt" in names
        assert "deep.txt" in names

    def test_find_withdirs(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/find-wd/sub", create_parents=True)
        dfs_fs.pipe_file(f"{d}/find-wd/file.txt", b"data")

        result = dfs_fs.find(f"{d}/find-wd", withdirs=True)
        # Should include the root dir, sub dir, and file
        assert f"{d}/find-wd" in result
        assert f"{d}/find-wd/sub" in result
        assert f"{d}/find-wd/file.txt" in result

    def test_find_detail(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/find-det")
        dfs_fs.pipe_file(f"{d}/find-det/f.txt", b"12345")

        result = dfs_fs.find(f"{d}/find-det", detail=True)
        assert isinstance(result, dict)
        info = result[f"{d}/find-det/f.txt"]
        assert info["type"] == "file"
        assert info["size"] == 5

    def test_find_maxdepth(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/find-md/l1/l2", create_parents=True)
        dfs_fs.pipe_file(f"{d}/find-md/top.txt", b"t")
        dfs_fs.pipe_file(f"{d}/find-md/l1/mid.txt", b"m")
        dfs_fs.pipe_file(f"{d}/find-md/l1/l2/deep.txt", b"d")

        result = dfs_fs.find(f"{d}/find-md", maxdepth=1)
        assert f"{d}/find-md/top.txt" in result
        assert f"{d}/find-md/l1/mid.txt" not in result

    def test_find_file_path(self, dfs_fs):
        """find on a file path returns just that file."""
        d = dfs_fs._test_dir
        dfs_fs.pipe_file(f"{d}/find-file.txt", b"data")
        result = dfs_fs.find(f"{d}/find-file.txt")
        assert result == [f"{d}/find-file.txt"]

    def test_find_not_found(self, dfs_fs):
        result = dfs_fs.find(f"{dfs_fs._test_dir}/nonexistent-dir")
        assert result == []

    def test_find_empty_dir(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/find-empty")
        result = dfs_fs.find(f"{d}/find-empty")
        assert result == []


@pytest.mark.live
class TestWalkEdgeCases:
    """walk() edge cases."""

    def test_walk_bottom_up(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.mkdir(f"{d}/bu/sub", create_parents=True)
        dfs_fs.pipe_file(f"{d}/bu/file.txt", b"data")

        results = list(dfs_fs.walk(f"{d}/bu", topdown=False))
        paths = [r[0] for r in results]
        # In bottom-up, children come before parents
        sub_idx = paths.index(f"{d}/bu/sub")
        root_idx = paths.index(f"{d}/bu")
        assert sub_idx < root_idx


@pytest.mark.live
class TestMakedirsEdgeCases:
    """makedirs() edge cases."""

    def test_makedirs_exist_ok_false_raises_on_file(self, dfs_fs):
        """makedirs(exist_ok=False) raises if a non-directory entry exists."""
        d = dfs_fs._test_dir
        # Create a file at the path, then try to makedirs there
        dfs_fs.pipe_file(f"{d}/not-a-dir", b"data")
        with pytest.raises(FileExistsError):
            dfs_fs.makedirs(f"{d}/not-a-dir/sub", exist_ok=False)


@pytest.mark.live
class TestTouchEdgeCases:
    """touch() edge cases."""

    def test_touch_nonexistent_no_truncate(self, dfs_fs):
        """touch(truncate=False) on nonexistent file should create it."""
        d = dfs_fs._test_dir
        path = f"{d}/touch-new.txt"
        dfs_fs.touch(path, truncate=False)
        assert dfs_fs.exists(path)


@pytest.mark.live
class TestAppendMode:
    """DFSFile append mode, against a real cluster.

    The note that used to be here; "DFS does not support true append --
    writes start at offset 0"; described a bug as a property of DAOS. DFS has
    no ``O_APPEND``, which is true and unrelated: appending at an offset we
    recorded ourselves needs no kernel flag. Writes landing at 0 overwrote the
    head of any existing file, and this class had only a new-file test, where
    offset 0 happens to be correct.

    ``tests/test_append_mode.py`` pins the offsets offline; these confirm the
    bytes actually land where expected on real DAOS.
    """

    def test_append_preserves_existing_content(self, dfs_fs):
        """The regression, end to end: the head of the file must survive.

        Offline tests assert the offset handed to dfs_write; this asserts the
        bytes that come back out, which is what a user actually loses.
        """
        d = dfs_fs._test_dir
        path = f"{d}/append-existing.txt"
        dfs_fs.pipe_file(path, b"HEAD-" * 10)
        original = dfs_fs.cat(path)

        with dfs_fs.open(path, "ab") as handle:
            handle.write(b"TAIL")

        result = dfs_fs.cat(path)
        assert result == original + b"TAIL", (
            "append did not preserve the original content; got "
            f"{result[:40]!r}... ({len(result)} bytes, expected "
            f"{len(original) + 4})"
        )

    def test_append_creates_new_file(self, dfs_fs):
        d = dfs_fs._test_dir
        path = f"{d}/append-new.txt"
        with dfs_fs.open(path, "ab") as f:
            f.write(b"new data")
        result = dfs_fs.cat(path)
        assert result == b"new data"


@pytest.mark.live
class TestStatusRepr:
    """status() and __repr__."""

    def test_status(self, dfs_fs):
        s = dfs_fs.status()
        assert "pool" in s
        assert "container" in s
        assert "closed" in s
        assert s["closed"] is False

    def test_repr(self, dfs_fs):
        r = repr(dfs_fs)
        assert "DFSFileSystem" in r


@pytest.mark.live
class TestSuggestOclassLive:
    """suggest_oclass() against real DAOS."""

    # Real DAOS 2.6 object-class IDs encode the class in the high bits, so they
    # exceed 0xFFFF (e.g. 0x01000001). Asserting `> 0xFFFF` catches the uint16
    # truncation bug live (it would return the low 16 bits, e.g. 1). The
    # deployment-independent guard is test_bindings::TestDfsSuggestOclass.
    def test_suggest_single(self, dfs_fs):
        oclass = dfs_fs.suggest_oclass("file:single")
        assert isinstance(oclass, int)
        assert oclass > 0xFFFF

    def test_suggest_max(self, dfs_fs):
        oclass = dfs_fs.suggest_oclass("file:max")
        assert isinstance(oclass, int)
        assert oclass > 0xFFFF

    def test_suggest_dir(self, dfs_fs):
        oclass = dfs_fs.suggest_oclass("dir:single")
        assert isinstance(oclass, int)
        assert oclass > 0xFFFF


@pytest.mark.live
class TestMountAttrLive:
    """mount_attr() against real DAOS."""

    def test_mount_attr(self, dfs_fs):
        attr = dfs_fs.mount_attr()
        assert hasattr(attr, "oclass_id")
        assert hasattr(attr, "chunk_size")
        assert hasattr(attr, "dir_oclass_id")
        # chunk_size is the meaningful mount-level default and must be non-zero
        # (1 MiB by default). This value assertion (not just hasattr) is what
        # locks the struct offset: if da_chunk_size were read at the wrong place
        # it would come back as 0 or garbage.
        assert attr.chunk_size > 0
        # The mount-level default oclass fields are legitimately 0 when the
        # container was created without explicit default object classes: DAOS
        # resolves the class per object at creation. So assert they are valid
        # non-negative ints, not that they are non-zero. (Per-object classes are
        # exercised with non-zero values in TestObjInfoLive.)
        assert attr.oclass_id >= 0
        assert attr.dir_oclass_id >= 0
        assert attr.file_oclass_id >= 0


@pytest.mark.live
class TestObjInfoLive:
    """obj_info() against real DAOS."""

    def test_obj_info_file(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.pipe_file(f"{d}/oinfo.txt", b"hello world")
        info = dfs_fs.obj_info(f"{d}/oinfo.txt")
        assert hasattr(info, "oclass_id")
        assert hasattr(info, "chunk_size")
        assert hasattr(info, "oid_hi")
        assert hasattr(info, "oid_lo")

        assert info.oclass_id > 0
        assert info.chunk_size > 0
        assert info.oid_hi != 0, "the high word carries class bits and cannot be zero"

    def test_obj_info_not_found(self, dfs_fs):
        with pytest.raises(FileNotFoundError):
            dfs_fs.obj_info(f"{dfs_fs._test_dir}/nonexistent.txt")


@pytest.mark.live
class TestXattrLive:
    """Extended attributes against real DAOS."""

    def test_setxattr_getxattr(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.pipe_file(f"{d}/xattr.txt", b"data")
        dfs_fs.setxattr(f"{d}/xattr.txt", "user.mykey", b"myvalue")
        val = dfs_fs.getxattr(f"{d}/xattr.txt", "user.mykey")
        assert val == b"myvalue"

    def test_listxattr(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.pipe_file(f"{d}/xlist.txt", b"data")
        dfs_fs.setxattr(f"{d}/xlist.txt", "user.k1", b"v1")
        dfs_fs.setxattr(f"{d}/xlist.txt", "user.k2", b"v2")
        names = dfs_fs.listxattr(f"{d}/xlist.txt")
        assert "user.k1" in names
        assert "user.k2" in names

    def test_removexattr(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.pipe_file(f"{d}/xrm.txt", b"data")
        dfs_fs.setxattr(f"{d}/xrm.txt", "user.todel", b"v")
        dfs_fs.removexattr(f"{d}/xrm.txt", "user.todel")
        names = dfs_fs.listxattr(f"{d}/xrm.txt")
        assert "user.todel" not in names


@pytest.mark.live
class TestHeadTailLive:
    """head() and tail() against real DAOS."""

    def test_head(self, dfs_fs):
        d = dfs_fs._test_dir
        data = b"0123456789abcdef"
        dfs_fs.pipe_file(f"{d}/head.txt", data)
        result = dfs_fs.head(f"{d}/head.txt", 5)
        assert result == data[:5]

    def test_tail(self, dfs_fs):
        d = dfs_fs._test_dir
        data = b"0123456789abcdef"
        dfs_fs.pipe_file(f"{d}/tail.txt", data)
        result = dfs_fs.tail(f"{d}/tail.txt", 5)
        assert result == data[-5:]


@pytest.mark.live
class TestPoolSpaceLive:
    """pool_space() against real DAOS."""

    def test_pool_space(self, dfs_fs):
        space = dfs_fs.pool_space()
        assert isinstance(space, B.PoolSpace)

        assert space.scm_total > 0
        assert 0 <= space.scm_free <= space.scm_total
        assert 0 <= space.nvme_free <= space.nvme_total


@pytest.mark.live
class TestChmodLive:
    """chmod() against real DAOS."""

    def test_chmod(self, dfs_fs):
        d = dfs_fs._test_dir
        dfs_fs.pipe_file(f"{d}/chmod.txt", b"data")
        dfs_fs.chmod(f"{d}/chmod.txt", 0o600)
        info = dfs_fs.info(f"{d}/chmod.txt")
        assert info["mode"] & 0o777 == 0o600


# Note: set_prefix is tested via mock only: it's a mount-level setting
# that corrupts path resolution for all subsequent operations on the DFS handle,
# making it unsafe to test in integration with other live tests.


# ── Mock-based tests for optimized methods (no DAOS needed) ─────────────


def _make_mock_fs(readonly=False):
    """Create a DFSFileSystem with all DAOS calls mocked.

    Returns (fs, mock_B, cleanup) where cleanup() must be called to unpatch.
    """
    import gc

    # Force GC before patching B: collects any orphaned DFSFile objects from
    # live tests while the real bindings are still active. Without this, GC
    # might collect them after B is patched, causing __del__ to call mock
    # dfs_close on a real (freed) DAOS handle → segfault.
    gc.collect()

    mock_B = MagicMock()
    # Constants
    mock_B.DAOS_PC_RO = 1
    mock_B.DAOS_PC_RW = 2
    mock_B.DAOS_COO_RO = 1
    mock_B.DAOS_COO_RW = 2

    mock_root_obj = MagicMock(name="root_obj")
    mock_root_stat = MagicMock(name="root_stat")
    mock_dfs_handle = MagicMock(name="dfs_handle")
    mock_poh = MagicMock(name="poh")
    mock_coh = MagicMock(name="coh")

    mock_B.daos_init = MagicMock()
    mock_B.pool_connect = MagicMock(return_value=mock_poh)
    mock_B.cont_open = MagicMock(return_value=mock_coh)
    mock_B.dfs_mount_fs = MagicMock(return_value=mock_dfs_handle)
    mock_B.dfs_lookup_path = MagicMock(return_value=(mock_root_obj, mock_root_stat))
    mock_B.dfs_release = MagicMock()
    mock_B.dfs_umount = MagicMock()
    mock_B.cont_close = MagicMock()
    mock_B.pool_disconnect = MagicMock()
    mock_B._CHUNK_SIZE = 16 * 1024 * 1024  # 16MB, same as real value
    mock_B._HAS_DFS_CONNECT = False  # Force low-level connect path in mock tests
    mock_B._HAS_PUNCH = True
    mock_B._HAS_SET_PREFIX = True
    mock_B._HAS_SUGGEST_OCLASS = True
    mock_B._ensure_libs = MagicMock()

    # DaosError must be a real exception class since _is_retryable uses isinstance()
    from daos_fsspec._bindings import (
        DER_HG as _DER_HG,
    )
    from daos_fsspec._bindings import (
        DER_NO_HDL as _DER_NO_HDL,
    )
    from daos_fsspec._bindings import (
        DER_TIMEDOUT as _DER_TIMEDOUT,
    )
    from daos_fsspec._bindings import (
        DER_UNREACH as _DER_UNREACH,
    )
    from daos_fsspec._bindings import (
        RETRYABLE_ERRORS as _RETRYABLE_ERRORS,
    )

    class _MockDaosError(Exception):
        def __init__(self, rc=0, message=""):
            self.rc = rc
            super().__init__(message)

        @property
        def is_retryable(self):
            return self.rc in _RETRYABLE_ERRORS

    mock_B.DaosError = _MockDaosError
    mock_B.DER_UNREACH = _DER_UNREACH
    mock_B.DER_HG = _DER_HG
    mock_B.DER_NO_HDL = _DER_NO_HDL
    mock_B.DER_TIMEDOUT = _DER_TIMEDOUT
    mock_B.RETRYABLE_ERRORS = _RETRYABLE_ERRORS

    patcher = patch_bindings(mock_B)
    patcher.__enter__()

    from daos_fsspec.core import DFSFileSystem

    fs = DFSFileSystem(pool="test-pool", container="test-cont", readonly=readonly)

    # Reset the root lookup mock so tests can set their own return values
    mock_B.dfs_lookup_path.reset_mock()

    def cleanup():
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass
        patcher.__exit__(None, None, None)

    return fs, mock_B, cleanup


class TestCatFile:
    """cat_file optimized path: direct lookup + read + release."""

    def test_uses_lookup_path(self):
        """cat_file uses dfs_lookup_path with full path, not _resolve_path."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock(name="file_obj")
            mock_stat = SimpleNamespace(size=13, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"hello, world!"

            result = fs.cat_file("/path/to/file.txt")

            assert result == b"hello, world!"
            mock_B.dfs_lookup_path.assert_called_once_with(
                fs._dfs, "/path/to/file.txt", os.O_RDONLY
            )
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 0, 13)
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_with_start_end(self):
        """cat_file respects start/end byte ranges."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"chunk"

            result = fs.cat_file("/file.bin", start=10, end=15)

            assert result == b"chunk"
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 10, 5)
        finally:
            cleanup()

    def test_negative_offsets(self):
        """cat_file resolves negative offsets using file size from stat."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"tail"

            result = fs.cat_file("/file.bin", start=-10)

            assert result == b"tail"
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 90, 10)
        finally:
            cleanup()

    def test_empty_range(self):
        """cat_file returns b'' when read_size is 0."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            result = fs.cat_file("/file.bin", start=50, end=50)

            assert result == b""
            mock_B.dfs_read.assert_not_called()
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_release_on_error(self):
        """dfs_release is called even when dfs_read raises."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.side_effect = RuntimeError("I/O error")

            with pytest.raises(RuntimeError, match="I/O error"):
                fs.cat_file("/file.bin")

            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_not_found(self):
        """FileNotFoundError propagates from dfs_lookup_path."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.side_effect = FileNotFoundError("no such file")

            with pytest.raises(FileNotFoundError):
                fs.cat_file("/missing.txt")
        finally:
            cleanup()

    def test_closed_fs(self):
        """cat_file raises RuntimeError on closed filesystem."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()

            with pytest.raises(RuntimeError, match="closed"):
                fs.cat_file("/file.txt")
        finally:
            cleanup()

    def test_full_file(self):
        """cat_file with no start/end reads entire file."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=5, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"hello"

            result = fs.cat_file("/file.txt")

            assert result == b"hello"
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 0, 5)
        finally:
            cleanup()


class TestCatRanges:
    """cat_ranges parallel reads."""

    def test_preserves_order(self):
        """Results match input path order, by content not by length.

        This used to assert only ``len(result) == 3`` and that each element was
        bytes, under a mock where ``dfs_lookup_path`` returned the *same* handle
        for every path and the payload was keyed on call order. Nothing in the
        fixture could tell which read belonged to which file, so reversing
        cat_ranges' output passed this test and the whole suite.

        Now each path gets its own handle and its own payload, so a result in
        the wrong order fails on the bytes.
        """
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            handles = {}

            def lookup(dfs, path, flags=0, *args, **kwargs):
                handle = handles.setdefault(path, MagicMock(name=f"obj:{path}"))
                return handle, SimpleNamespace(size=100, mode=0o100644, is_dir=False)

            by_handle = {}

            def read(dfs, obj, offset, size):
                return by_handle[id(obj)]

            mock_B.dfs_lookup_path.side_effect = lookup
            mock_B.dfs_read.side_effect = read

            paths = ["/alpha", "/beta", "/gamma"]
            for path in paths:
                handle = handles.setdefault(path, MagicMock(name=f"obj:{path}"))
                by_handle[id(handle)] = path.encode()

            result = fs.cat_ranges(paths, [0, 0, 0], [5, 5, 5])

            assert result == [b"/alpha", b"/beta", b"/gamma"], (
                f"cat_ranges returned results out of order: {result}"
            )
        finally:
            cleanup()

    def test_on_error_return(self):
        """on_error='return' collects exceptions in result list."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            # Use cat_file mock to control which path fails
            _orig_cat_file = fs.cat_file

            def fake_cat_file(path, start=None, end=None, **kwargs):
                if path == "/b":
                    raise RuntimeError("read failed")
                return b"ok"

            with patch.object(fs, "cat_file", side_effect=fake_cat_file):
                result = fs.cat_ranges(["/a", "/b", "/c"], [0, 0, 0], [5, 5, 5], on_error="return")

            assert len(result) == 3
            assert result[0] == b"ok"
            assert isinstance(result[1], RuntimeError)
            assert result[2] == b"ok"
        finally:
            cleanup()

    def test_on_error_raise(self):
        """on_error='raise' propagates the first exception."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            call_count = [0]

            def fake_read(dfs, obj, offset, size):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("read failed")
                return b"ok"

            mock_B.dfs_read.side_effect = fake_read

            with pytest.raises(RuntimeError, match="read failed"):
                fs.cat_ranges(["/a", "/b"], [0, 0], [5, 5], on_error="raise")
        finally:
            cleanup()

    def test_broadcast_scalars(self):
        """Single int start/end is broadcast to all paths."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"data"

            fs.cat_ranges(["/a", "/b", "/c"], 0, 10)

            assert mock_B.dfs_read.call_count == 3
            for call in mock_B.dfs_read.call_args_list:
                assert call.args == (fs._dfs, mock_obj, 0, 10)
        finally:
            cleanup()

    def test_type_validation(self):
        """cat_ranges raises TypeError if paths is not a list."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(TypeError, match="paths must be a list"):
                fs.cat_ranges("/single/path", 0, 5)
        finally:
            cleanup()

    def test_length_validation(self):
        """starts/ends lists must match paths length."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(ValueError, match="matching paths length"):
                fs.cat_ranges(["/a", "/b"], [0], [5, 5])
            with pytest.raises(ValueError, match="matching paths length"):
                fs.cat_ranges(["/a", "/b"], [0, 0], [5])
        finally:
            cleanup()

    def test_on_error_validation(self):
        """on_error must be one of the supported modes."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(ValueError, match="on_error"):
                fs.cat_ranges(["/a"], [0], [5], on_error="ignore")
        finally:
            cleanup()

    def test_single_path_no_threadpool(self):
        """Single path doesn't create a ThreadPoolExecutor."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=10, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"data"

            with patch("concurrent.futures.ThreadPoolExecutor") as mock_pool:
                result = fs.cat_ranges(["/a"], [0], [4])
                mock_pool.assert_not_called()

            assert result == [b"data"]
        finally:
            cleanup()


def _make_mock_dir_entry(name, is_dir=False, size=0, mode=0o100644):
    """Create a mock DirEntry with stat info."""
    stat = SimpleNamespace(
        size=size,
        mode=mode,
        mtime=1700000000,
        atime=1700000000,
        uid=1000,
        gid=1000,
        is_dir=is_dir,
        type_name="directory" if is_dir else "file",
    )
    entry = SimpleNamespace(name=name, stat=stat)
    return entry


class TestFind:
    """find() readdirplus-batched recursive listing."""

    def _dir_stat(size=0):
        """Create a mock stat for a directory (with type_name)."""
        return SimpleNamespace(
            size=size,
            is_dir=True,
            type_name="directory",
            mode=0o40755,
            mtime=1700000000,
            atime=1700000000,
            uid=1000,
            gid=1000,
        )

    def test_flat_directory(self):
        """find returns sorted file paths for a flat directory."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            entries = [
                _make_mock_dir_entry("c.txt", size=30),
                _make_mock_dir_entry("a.txt", size=10),
                _make_mock_dir_entry("b.txt", size=20),
            ]
            mock_obj = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_obj, TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = entries

            result = fs.find("/mydir")

            assert result == ["/mydir/a.txt", "/mydir/b.txt", "/mydir/c.txt"]
        finally:
            cleanup()

    def test_nested_directory(self):
        """find recurses into subdirectories."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            dir_entries = [
                _make_mock_dir_entry("file.txt", size=10),
                _make_mock_dir_entry("sub", is_dir=True),
            ]
            sub_entries = [
                _make_mock_dir_entry("inner.txt", size=20),
            ]

            mock_obj = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_obj, TestFind._dir_stat())

            call_count = [0]

            def fake_readdir(dfs, obj):
                call_count[0] += 1
                if call_count[0] == 1:
                    return dir_entries
                return sub_entries

            mock_B.dfs_readdirplus.side_effect = fake_readdir

            result = fs.find("/mydir")

            assert "/mydir/file.txt" in result
            assert "/mydir/sub/inner.txt" in result
            assert len(result) == 2
        finally:
            cleanup()

    def test_withdirs(self):
        """find includes directory entries when withdirs=True."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            # /mydir has a file and a subdir; /mydir/subdir has only a file
            dir_entries = [
                _make_mock_dir_entry("file.txt", size=10),
                _make_mock_dir_entry("subdir", is_dir=True),
            ]
            subdir_entries = [
                _make_mock_dir_entry("inner.txt", size=5),
            ]

            mock_obj = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_obj, TestFind._dir_stat())

            call_count = [0]

            def fake_readdir(dfs, obj):
                call_count[0] += 1
                if call_count[0] == 1:
                    return dir_entries
                return subdir_entries

            mock_B.dfs_readdirplus.side_effect = fake_readdir

            def fake_info(path):
                if path == "/mydir":
                    return {"name": "/mydir", "size": 0, "type": "directory"}
                if path == "/mydir/subdir":
                    return {"name": "/mydir/subdir", "size": 0, "type": "directory"}
                return {"name": path, "size": 0, "type": "file"}

            with patch.object(fs, "info", side_effect=fake_info):
                result = fs.find("/mydir", withdirs=True)

            assert "/mydir" in result
            assert "/mydir/subdir" in result
            assert "/mydir/file.txt" in result
            assert "/mydir/subdir/inner.txt" in result
        finally:
            cleanup()

    def test_detail(self):
        """find returns detail dict when detail=True."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            entries = [
                _make_mock_dir_entry("file.txt", size=42, mode=0o100644),
            ]
            mock_obj = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_obj, TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = entries

            result = fs.find("/mydir", detail=True)

            assert isinstance(result, dict)
            info = result["/mydir/file.txt"]
            assert info["name"] == "/mydir/file.txt"
            assert info["size"] == 42
            assert info["type"] == "file"
            assert info["mode"] == 0o100644
            assert "mtime" in info
        finally:
            cleanup()

    def test_maxdepth(self):
        """find respects maxdepth limit."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            dir_entries = [
                _make_mock_dir_entry("file.txt", size=10),
                _make_mock_dir_entry("sub", is_dir=True),
            ]
            sub_entries = [
                _make_mock_dir_entry("inner.txt", size=20),
            ]

            mock_obj = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_obj, TestFind._dir_stat())

            call_count = [0]

            def fake_readdir(dfs, obj):
                call_count[0] += 1
                if call_count[0] == 1:
                    return dir_entries
                return sub_entries

            mock_B.dfs_readdirplus.side_effect = fake_readdir

            result = fs.find("/mydir", maxdepth=1)

            assert result == ["/mydir/file.txt"]
            assert mock_B.dfs_readdirplus.call_count == 1
        finally:
            cleanup()

    def test_file_path(self):
        """find returns just the file when path is a file, not a directory."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:

            def fake_info(path):
                if path == "/mydir/file.txt":
                    return {"name": "/mydir/file.txt", "size": 100, "type": "file"}

            with patch.object(fs, "info", side_effect=fake_info):
                result = fs.find("/mydir/file.txt")

            assert result == ["/mydir/file.txt"]
            mock_B.dfs_readdirplus.assert_not_called()
        finally:
            cleanup()

    def test_root_path(self):
        """find works on root path using self._root directly."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            entries = [
                _make_mock_dir_entry("file.txt", size=10),
            ]
            mock_B.dfs_readdirplus.return_value = entries

            with patch.object(
                fs, "info", return_value={"name": "/", "size": 0, "type": "directory"}
            ):
                result = fs.find("/")

            assert result == ["/file.txt"]
        finally:
            cleanup()

    def test_not_found(self):
        """find returns empty for non-existent path."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with patch.object(fs, "info", side_effect=FileNotFoundError):
                result = fs.find("/nonexistent")

            assert result == []
        finally:
            cleanup()

    def test_closed_fs(self):
        """find raises RuntimeError on closed filesystem."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()

            with pytest.raises(RuntimeError, match="closed"):
                fs.find("/path")
        finally:
            cleanup()


# ── Edge case and error handling tests (mock-based) ─────────────────────


class TestInitValidation:
    """DFSFileSystem.__init__ validation."""

    def test_empty_pool_raises(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError):
            DFSFileSystem(pool="", container="test")

    def test_empty_container_raises(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError):
            DFSFileSystem(pool="test", container="")

    def test_label_too_long_raises(self):
        from daos_fsspec.core import DFSFileSystem

        with pytest.raises(ValueError):
            DFSFileSystem(pool="a" * 200, container="test")


class TestHealthCheckClosed:
    """health_check on closed FS returns False."""

    def test_health_check_closed(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            assert fs.health_check() is False
        finally:
            cleanup()


class TestCloseIdempotent:
    """close() is idempotent."""

    def test_double_close_no_error(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            assert fs._closed
            # Second close should be a no-op
            fs.close()
            assert fs._closed
        finally:
            cleanup()


class TestRetryReconnect:
    """_retry reconnect path on DER_UNREACH."""

    def test_retry_reconnects_on_der_unreach(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            call_count = [0]

            def flaky():
                call_count[0] += 1
                if call_count[0] <= 1:
                    raise mock_B.DaosError(mock_B.DER_UNREACH, "unreachable")
                return "ok"

            result = fs._retry(flaky)
            assert result == "ok"
            assert call_count[0] == 2
        finally:
            cleanup()

    def test_retry_sets_closed_on_reconnect_failure(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.side_effect = mock_B.DaosError(mock_B.DER_UNREACH, "still down")

            def always_fail():
                raise mock_B.DaosError(mock_B.DER_UNREACH, "fail")

            # Make reconnect also fail
            mock_B.pool_connect.side_effect = OSError("connect failed")

            with pytest.raises(mock_B.DaosError):
                fs._retry(always_fail)
        finally:
            cleanup()


class TestIsRetryableOSError:
    """_is_retryable for OSError with retryable errno."""

    def test_retryable_errno(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            import errno

            exc = OSError(errno.EAGAIN, "try again")
            assert fs._is_retryable(exc) is True
        finally:
            cleanup()

    def test_non_retryable_errno(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            exc = OSError(2, "no such file")
            assert fs._is_retryable(exc) is False
        finally:
            cleanup()


class TestPipeFileReadonly:
    """pipe_file on readonly FS raises PermissionError."""

    def test_pipe_file_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.pipe_file("/file", b"data")
        finally:
            cleanup()


class TestReadonlyGuards:
    """Mutating ops on a read-only FS raise PermissionError before touching libdfs."""

    def test_mkdir_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.mkdir("/d")
            mock_B.dfs_mkdir.assert_not_called()
        finally:
            cleanup()

    def test_makedirs_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.makedirs("/a/b", exist_ok=True)
        finally:
            cleanup()

    def test_touch_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.touch("/f")
        finally:
            cleanup()

    def test_symlink_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.symlink("/link", "/target")
            mock_B.dfs_create_symlink.assert_not_called()
        finally:
            cleanup()

    def test_open_write_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.open("/f", "wb")
            mock_B.dfs_open_file.assert_not_called()
        finally:
            cleanup()


class TestMakedirsConformance:
    """makedirs(exist_ok=False) raises FileExistsError on an existing directory."""

    def test_makedirs_existing_no_exist_ok_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with patch.object(fs, "exists", return_value=True):
                with pytest.raises(FileExistsError):
                    fs.makedirs("/already/there", exist_ok=False)
        finally:
            cleanup()

    def test_makedirs_existing_exist_ok_ok(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with (
                patch.object(fs, "exists", return_value=True),
                patch.object(fs, "mkdir") as mock_mkdir,
            ):
                # exist_ok=True must not raise even when the path exists.
                fs.makedirs("/already/there", exist_ok=True)
                mock_mkdir.assert_called_once()
        finally:
            cleanup()


class TestDfsReadNoTruncation:
    """dfs_read must keep reading after a short (non-EOF) read and only stop on empty."""

    def test_short_read_continues_until_empty(self):
        import daos_fsspec._bindings as B

        chunk = B._CHUNK_SIZE
        # First single read returns a short chunk (not EOF), second returns the
        # rest, third returns empty (EOF). Total size requested spans 2 chunks.
        reads = [b"a" * (chunk // 2), b"b" * chunk, b""]

        with (
            patch.object(B, "_ensure_libs", return_value=(MagicMock(), MagicMock())),
            patch.object(B, "_dfs_read_single", side_effect=reads) as m,
        ):
            out = B.dfs_read(MagicMock(), MagicMock(), 0, chunk * 2)

        # Must not stop at the first short read: all non-empty data returned.
        assert out == reads[0] + reads[1]
        assert m.call_count == 3


class TestLockReentrancy:
    """The mount lock is reentrant so handle-release helpers are safe under it."""

    def test_lock_is_reentrant(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            # A non-reentrant Lock would deadlock on the second acquire.
            with fs._lock:
                with fs._lock:
                    pass
        finally:
            cleanup()

    def test_release_parent_under_held_lock(self):
        # _release_parent acquires the lock internally; calling it while the
        # lock is already held must not deadlock (regression for RLock change).
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            sentinel = MagicMock(name="parent_obj")
            with fs._lock:
                fs._release_parent(sentinel)
            mock_B.dfs_release.assert_called_once_with(sentinel)
        finally:
            cleanup()


class TestMkdirDepthLimit:
    """mkdir depth limit."""

    def test_mkdir_too_deep(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            deep = "/".join([f"d{i}" for i in range(101)])
            with pytest.raises(ValueError, match="exceeds max"):
                fs.mkdir(f"/{deep}", create_parents=True)
        finally:
            cleanup()


class TestCheckErrorMapping:
    """_check error mapping from DAOS errno to Python exceptions."""

    def test_check_eacces_raises_permission_error(self):
        import errno as errno_mod

        from daos_fsspec._bindings import _check

        with pytest.raises(PermissionError):
            _check(errno_mod.EACCES, "test operation")

    def test_check_enotdir_raises_not_a_directory(self):
        import errno as errno_mod

        from daos_fsspec._bindings import _check

        with pytest.raises(NotADirectoryError):
            _check(errno_mod.ENOTDIR, "test operation")

    def test_check_eisdir_raises_is_a_directory(self):
        import errno as errno_mod

        from daos_fsspec._bindings import _check

        with pytest.raises(IsADirectoryError):
            _check(errno_mod.EISDIR, "test operation")

    def test_check_enotempty_raises_oserror(self):
        import errno as errno_mod

        from daos_fsspec._bindings import _check

        with pytest.raises(OSError):
            _check(errno_mod.ENOTEMPTY, "test operation")

    def test_check_eexist_raises_file_exists(self):
        import errno as errno_mod

        from daos_fsspec._bindings import _check

        with pytest.raises(FileExistsError):
            _check(errno_mod.EEXIST, "test operation")


# ── Pickling ───────────────────────────────────────────────────────────


class TestPickling:
    """DFSFileSystem should raise a clear error when pickled."""

    def test_reduce_is_our_own_override(self):
        """``__reduce__`` must be ours, not object's."""
        from daos_fsspec.core import DFSFileSystem

        # Not `is not object.__reduce__`: fsspec's AbstractFileSystem defines
        # its own, which is what a pickled filesystem would fall back to --
        # deleting ours left that comparison passing. Ask who implements it.
        owner = DFSFileSystem.__reduce__.__module__
        assert owner.startswith("daos_fsspec"), (
            f"DFSFileSystem.__reduce__ resolves to {owner}; our refusal to "
            "pickle was removed, so fsspec's would silently reconstruct a "
            "filesystem carrying dead DAOS handles"
        )


# ── struct_stat validation ─────────────────────────────────────────────


class TestStructStat:
    """struct_stat matches the platform's struct stat."""

    def test_struct_stat_size(self):
        """struct_stat size should match platform's struct stat."""
        import platform

        from daos_fsspec._bindings import sizeof, struct_stat

        arch = platform.machine()
        size = sizeof(struct_stat)

        expected_sizes = {
            "x86_64": 144,
            "aarch64": 144,
            "ppc64le": 144,
            "s390x": 144,
        }

        if arch in expected_sizes:
            assert size == expected_sizes[arch], (
                f"struct_stat size {size} != expected {expected_sizes[arch]} on {arch}"
            )
        else:
            assert 96 <= size <= 256, f"struct_stat size {size} looks wrong for {arch}"

    def test_struct_stat_field_offsets(self):
        """Verify critical field offsets match platform expectations."""
        from daos_fsspec._bindings import struct_stat

        # st_mode should be at offset 24 on x86_64/aarch64 glibc
        # st_dev(8) + st_ino(8) + st_nlink(8) = 24
        mode_field = getattr(struct_stat, "st_mode")
        assert mode_field.offset == 24, f"st_mode offset {mode_field.offset} != 24"

        # st_size should be at offset 48
        # st_dev(8) + st_ino(8) + st_nlink(8) + st_mode(4) + st_uid(4) + st_gid(4) + __pad0(4) + st_rdev(8) = 48
        size_field = getattr(struct_stat, "st_size")
        assert size_field.offset == 48, f"st_size offset {size_field.offset} != 48"


# ── Checksum (mock-based) ─────────────────────────────────────────────


class TestChecksum:
    """checksum() for cache invalidation."""

    def test_checksum_returns_string(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(
                size=100, mode=0o100644, mtime=1700000000.123456, is_dir=False
            )
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            result = fs.checksum("/file.txt")

            assert isinstance(result, str)
            assert "33188" in result  # 0o100644 in decimal
            assert "100" in result
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_checksum_not_found(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.side_effect = FileNotFoundError("no such file")

            with pytest.raises(FileNotFoundError):
                fs.checksum("/missing.txt")
        finally:
            cleanup()

    def test_checksum_closed_fs(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()

            with pytest.raises(RuntimeError, match="closed"):
                fs.checksum("/file.txt")
        finally:
            cleanup()


# ── du (mock-based) ───────────────────────────────────────────────────


class TestDu:
    """du() disk usage."""

    def test_du_single_file(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:

            def fake_info(path):
                return {"name": path, "size": 1000, "type": "file"}

            with patch.object(fs, "info", side_effect=fake_info):
                result = fs.du("/file.txt")

            assert result == {"/file.txt": 1000}
        finally:
            cleanup()

    def test_du_single_file_total(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:

            def fake_info(path):
                return {"name": path, "size": 1000, "type": "file"}

            with patch.object(fs, "info", side_effect=fake_info):
                result = fs.du("/file.txt", total=True)

            assert result == 1000
        finally:
            cleanup()

    def test_du_directory(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            entries = [
                _make_mock_dir_entry("a.txt", size=100),
                _make_mock_dir_entry("b.txt", size=200),
            ]
            mock_obj = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_obj, TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = entries

            with patch.object(
                fs, "info", return_value={"name": "/dir", "size": 0, "type": "directory"}
            ):
                result = fs.du("/dir")

            assert sum(result.values()) == 300
        finally:
            cleanup()

    def test_du_not_found(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with patch.object(fs, "info", side_effect=FileNotFoundError):
                result = fs.du("/nonexistent")

            assert result == {}
        finally:
            cleanup()


# ── get_file / put_file (mock-based) ──────────────────────────────────


class TestStreamingTransfer:
    """get_file/put_file streaming transfer."""

    def test_get_file(self):
        import tempfile

        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=5, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"hello"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp_path = tmp.name

            try:
                fs.get_file("/remote/file.txt", tmp_path)
                with open(tmp_path, "rb") as f:
                    assert f.read() == b"hello"
                mock_B.dfs_release.assert_called_once_with(mock_obj)
            finally:
                os.unlink(tmp_path)
        finally:
            cleanup()

    def test_put_file(self):
        import tempfile

        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_parent = MagicMock()
            mock_B.dfs_lookup_path.return_value = (mock_parent, SimpleNamespace())
            mock_B.dfs_lookup_rel_path.return_value = (
                mock_parent,
                SimpleNamespace(),
            )
            mock_B.dfs_open_file.return_value = mock_obj

            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp.write(b"hello")
                tmp_path = tmp.name

            try:
                fs.put_file(tmp_path, "/remote/file.txt")
                mock_B.dfs_open_file.assert_called_once()
                mock_B.dfs_write.assert_called_once()
                # Verify the data written is the full file content
                call_args = mock_B.dfs_write.call_args
                assert call_args[0][3] == b"hello"  # data argument
                mock_B.dfs_close.assert_called_once_with(mock_obj)
            finally:
                os.unlink(tmp_path)
        finally:
            cleanup()

    def test_put_file_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.put_file("/local/file.txt", "/remote/file.txt")
        finally:
            cleanup()


# ── xattr operations (mock-based) ────────────────────────────────────


class TestXattr:
    """Extended attribute operations (getxattr, setxattr, listxattr, removexattr)."""

    def test_getxattr(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_getxattr.return_value = b"abc123"

            result = fs.getxattr("/file.txt", "user.md5")

            assert result == b"abc123"
            mock_B.dfs_getxattr.assert_called_once_with(fs._dfs, mock_obj, "user.md5")
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_getxattr_not_found(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_getxattr.side_effect = FileNotFoundError("no such xattr")

            with pytest.raises(FileNotFoundError):
                fs.getxattr("/file.txt", "user.nonexistent")
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_setxattr(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            fs.setxattr("/file.txt", "user.md5", b"abc123")

            mock_B.dfs_setxattr.assert_called_once_with(fs._dfs, mock_obj, "user.md5", b"abc123", 0)
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_setxattr_with_flags(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            from daos_fsspec._bindings import XATTR_CREATE

            fs.setxattr("/file.txt", "user.new", b"val", flags=XATTR_CREATE)

            mock_B.dfs_setxattr.assert_called_once_with(
                fs._dfs, mock_obj, "user.new", b"val", XATTR_CREATE
            )
        finally:
            cleanup()

    def test_setxattr_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.setxattr("/file.txt", "user.x", b"v")
        finally:
            cleanup()

    def test_listxattr(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_listxattr.return_value = ["user.md5", "user.sha256"]

            result = fs.listxattr("/file.txt")

            assert result == ["user.md5", "user.sha256"]
            mock_B.dfs_listxattr.assert_called_once_with(fs._dfs, mock_obj)
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_listxattr_empty(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_listxattr.return_value = []

            result = fs.listxattr("/file.txt")

            assert result == []
        finally:
            cleanup()

    def test_removexattr(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            fs.removexattr("/file.txt", "user.md5")

            mock_B.dfs_removexattr.assert_called_once_with(fs._dfs, mock_obj, "user.md5")
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_removexattr_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.removexattr("/file.txt", "user.x")
        finally:
            cleanup()

    def test_getxattr_closed_fs(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.getxattr("/file.txt", "user.x")
        finally:
            cleanup()


# ── head / tail (mock-based) ─────────────────────────────────────────


class TestHeadTail:
    """head() and tail() optimized reads via cat_file range."""

    def test_head(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"first10"

            result = fs.head("/file.txt", size=7)

            assert result == b"first10"
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 0, 7)
        finally:
            cleanup()

    def test_head_default_size(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=10000, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"x" * 1024

            result = fs.head("/file.txt")

            assert len(result) == 1024
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 0, 1024)
        finally:
            cleanup()

    def test_tail(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_read.return_value = b"last20"

            result = fs.tail("/file.txt", size=20)

            assert result == b"last20"
            mock_B.dfs_read.assert_called_once_with(fs._dfs, mock_obj, 80, 20)
        finally:
            cleanup()

    def test_negative_size_rejected(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(ValueError, match="size"):
                fs.head("/file.txt", size=-1)
            with pytest.raises(ValueError, match="size"):
                fs.tail("/file.txt", size=-1)
        finally:
            cleanup()


# ── chmod (mock-based) ───────────────────────────────────────────────


class TestChmod:
    """chmod() permission changes."""

    def test_chmod(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_parent = MagicMock()
            mock_stat = SimpleNamespace(size=0, mode=0o100644, is_dir=False, is_link=False)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)

            fs.chmod("/path/to/file.txt", 0o755)

            mock_B.dfs_chmod.assert_called_once_with(fs._dfs, mock_parent, "file.txt", 0o755)
        finally:
            cleanup()

    def test_chmod_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.chmod("/file.txt", 0o755)
        finally:
            cleanup()

    def test_chmod_closed_fs(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.chmod("/file.txt", 0o755)
        finally:
            cleanup()


# ── pool_space (mock-based) ──────────────────────────────────────────


class TestPoolSpace:
    """pool_space() storage usage query."""

    def test_pool_space(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            from daos_fsspec._bindings import PoolSpace

            mock_poh = MagicMock()
            mock_B.dfs_pool_get.return_value = mock_poh
            mock_B.pool_query_space.return_value = PoolSpace(
                scm_total=1024 * 1024 * 1024,
                scm_free=512 * 1024 * 1024,
                nvme_total=4 * 1024 * 1024 * 1024,
                nvme_free=2 * 1024 * 1024 * 1024,
            )

            result = fs.pool_space()

            assert result.scm_total == 1024 * 1024 * 1024
            assert result.scm_free == 512 * 1024 * 1024
            assert result.nvme_total == 4 * 1024 * 1024 * 1024
            assert result.nvme_free == 2 * 1024 * 1024 * 1024
            mock_B.dfs_pool_get.assert_called_once_with(fs._dfs)
            mock_B.dfs_pool_put.assert_called_once_with(fs._dfs, mock_poh)
            mock_B.pool_query_space.assert_called_once_with(mock_poh)
        finally:
            cleanup()

    def test_pool_space_closed_fs(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.pool_space()
        finally:
            cleanup()


# ── object_class / chunk_size params ─────────────────────────────────


class TestObjectClassChunkSize:
    """Verify object_class and chunk_size params are passed through."""

    def test_init_stores_params(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            # Default params
            assert fs._object_class == 0
            assert fs._chunk_size == 0
        finally:
            cleanup()

    def test_init_custom_params(self):
        mock_B = MagicMock()
        mock_B.DAOS_PC_RO = 1
        mock_B.DAOS_PC_RW = 2
        mock_B.DAOS_COO_RO = 1
        mock_B.DAOS_COO_RW = 2

        mock_root_obj = MagicMock(name="root_obj")
        mock_root_stat = MagicMock(name="root_stat")
        mock_dfs_handle = MagicMock(name="dfs_handle")
        mock_poh = MagicMock(name="poh")
        mock_coh = MagicMock(name="coh")

        mock_B.daos_init = MagicMock()
        mock_B.pool_connect = MagicMock(return_value=mock_poh)
        mock_B.cont_open = MagicMock(return_value=mock_coh)
        mock_B.dfs_mount_fs = MagicMock(return_value=mock_dfs_handle)
        mock_B.dfs_lookup_path = MagicMock(return_value=(mock_root_obj, mock_root_stat))
        mock_B.dfs_release = MagicMock()
        mock_B.dfs_umount = MagicMock()
        mock_B.cont_close = MagicMock()
        mock_B.pool_disconnect = MagicMock()
        mock_B._CHUNK_SIZE = 16 * 1024 * 1024
        mock_B._HAS_DFS_CONNECT = False

        from daos_fsspec._bindings import RETRYABLE_ERRORS as _RETRYABLE_ERRORS

        class _MockDaosError(Exception):
            def __init__(self, rc=0, message=""):
                self.rc = rc
                super().__init__(message)

            @property
            def is_retryable(self):
                return self.rc in _RETRYABLE_ERRORS

        mock_B.DaosError = _MockDaosError

        patcher = patch_bindings(mock_B)
        patcher.__enter__()

        try:
            from daos_fsspec.core import DFSFileSystem

            fs = DFSFileSystem(
                pool="test-pool",
                container="test-cont",
                object_class=7,
                chunk_size=4 * 1024 * 1024,
            )
            assert fs._object_class == 7
            assert fs._chunk_size == 4 * 1024 * 1024

            # Verify the status includes the new fields
            s = fs.status()
            assert s["object_class"] == 7
            assert s["chunk_size"] == 4 * 1024 * 1024

            fs.close()
        finally:
            patcher.__exit__(None, None, None)

    def test_pipe_file_passes_oclass_and_chunk(self):
        """pipe_file should pass object_class and chunk_size to dfs_open_file."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            # Set custom oclass/chunk on the filesystem
            fs._object_class = 7
            fs._chunk_size = 4 * 1024 * 1024

            mock_parent = MagicMock()
            mock_stat = SimpleNamespace()
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)

            mock_obj = MagicMock()
            mock_B.dfs_open_file.return_value = mock_obj

            fs.pipe_file("/path/to/file.txt", b"hello")

            # Verify the open call includes oclass and chunk_size
            call_args = mock_B.dfs_open_file.call_args
            assert call_args[1].get("oclass_id") == 7
            assert call_args[1].get("chunk_size") == 4 * 1024 * 1024
        finally:
            cleanup()


# ── status() includes new fields ─────────────────────────────────────


class TestStatusNewFields:
    """status() includes object_class and chunk_size."""

    def test_status_has_new_fields(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            s = fs.status()
            assert "object_class" in s
            assert "chunk_size" in s
            assert s["object_class"] == 0
            assert s["chunk_size"] == 0
        finally:
            cleanup()


# ── pool passes object_class and chunk_size ──────────────────────────


class TestPoolObjectClassChunkSize:
    """DFSFileSystemPool passes object_class/chunk_size to connections."""

    def test_pool_passes_params(self):
        from daos_fsspec.pool import DFSFileSystemPool

        with patch("daos_fsspec.pool.DFSFileSystem") as mock_cls:
            mock_cls.return_value = MagicMock()
            DFSFileSystemPool(
                pool="test-pool",
                container="test-cont",
                size=2,
                object_class=7,
                chunk_size=4 * 1024 * 1024,
            )
            assert mock_cls.call_count == 2
            for call in mock_cls.call_args_list:
                assert call[1]["object_class"] == 7
                assert call[1]["chunk_size"] == 4 * 1024 * 1024


# ── chown (mock-based) ──────────────────────────────────────────────


class TestChown:
    """chown() ownership changes."""

    def test_chown(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_parent = MagicMock()
            mock_stat = SimpleNamespace(size=0, mode=0o100644, is_dir=False, is_link=False)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)

            fs.chown("/path/to/file.txt", uid=1000, gid=1000)

            mock_B.dfs_chown.assert_called_once_with(
                fs._dfs, mock_parent, "file.txt", 1000, 1000, 0
            )
        finally:
            cleanup()

    def test_chown_no_follow(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_parent = MagicMock()
            mock_stat = SimpleNamespace(size=0, mode=0o100644, is_dir=False, is_link=False)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)

            fs.chown("/path/to/file.txt", uid=1000, gid=1000, follow_links=False)

            from daos_fsspec.core import _O_NOFOLLOW

            mock_B.dfs_chown.assert_called_once_with(
                fs._dfs, mock_parent, "file.txt", 1000, 1000, _O_NOFOLLOW
            )
        finally:
            cleanup()

    def test_chown_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.chown("/file.txt", uid=1000, gid=1000)
        finally:
            cleanup()


# ── access (mock-based) ─────────────────────────────────────────────


class TestAccess:
    """access() permission check."""

    def test_access_granted(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_access.return_value = True

            result = fs.access("/file.txt", mode=0)

            assert result is True
            # For a single-component path, _resolve_path returns (self._root, "file.txt")
            mock_B.dfs_access.assert_called_once()
            call_args = mock_B.dfs_access.call_args[0]
            assert call_args[2] == "file.txt"
            assert call_args[3] == 0
        finally:
            cleanup()

    def test_access_denied(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_access.return_value = False

            result = fs.access("/file.txt", mode=2)  # W_OK

            assert result is False
        finally:
            cleanup()

    def test_access_root(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            result = fs.access("/", mode=0)
            assert result is True
        finally:
            cleanup()


# ── mount_attr (mock-based) ─────────────────────────────────────────


class TestMountAttr:
    """mount_attr() DFS mount attribute query."""

    def test_mount_attr(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            from daos_fsspec._bindings import MountAttr

            mock_B.dfs_query_attr.return_value = MountAttr(
                oclass_id=0,
                dir_oclass_id=0,
                file_oclass_id=0,
                chunk_size=1048576,
                mode=0,
            )

            result = fs.mount_attr()

            assert result.chunk_size == 1048576
            mock_B.dfs_query_attr.assert_called_once_with(fs._dfs)
        finally:
            cleanup()

    def test_mount_attr_closed(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.mount_attr()
        finally:
            cleanup()


# ── obj_info (mock-based) ───────────────────────────────────────────


class TestObjInfo:
    """obj_info() query object class, chunk_size, OID."""

    def test_obj_info(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            from daos_fsspec._bindings import ObjInfo

            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)
            mock_B.dfs_obj_get_info.return_value = ObjInfo(
                oclass_id=7,
                dir_oclass_id=0,
                file_oclass_id=0,
                chunk_size=4194304,
                oid_hi=1,
                oid_lo=2,
            )

            result = fs.obj_info("/file.txt")

            assert result.oclass_id == 7
            assert result.chunk_size == 4194304
            # Pin the OID words too, so the plumbing direction (hi vs lo) is
            # locked at this layer and can't silently swap.
            assert result.oid_hi == 1
            assert result.oid_lo == 2
            mock_B.dfs_obj_get_info.assert_called_once_with(fs._dfs, mock_obj)
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_obj_info_not_found(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.side_effect = FileNotFoundError("no such file")

            with pytest.raises(FileNotFoundError):
                fs.obj_info("/missing.txt")
        finally:
            cleanup()


# ── truncate (mock-based) ───────────────────────────────────────────


class TestTruncate:
    """truncate() via dfs_punch."""

    def test_truncate(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_obj = MagicMock()
            mock_stat = SimpleNamespace(size=100, mode=0o100644, is_dir=False)
            mock_B.dfs_lookup_path.return_value = (mock_obj, mock_stat)

            fs.truncate("/file.txt", size=50)

            mock_B.dfs_punch.assert_called_once()
            call_args = mock_B.dfs_punch.call_args
            assert call_args[0][1] == mock_obj  # obj
            assert call_args[0][2] == 50  # offset
            mock_B.dfs_release.assert_called_once_with(mock_obj)
        finally:
            cleanup()

    def test_truncate_readonly(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.truncate("/file.txt", size=0)
        finally:
            cleanup()

    def test_truncate_closed(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.truncate("/file.txt", size=0)
        finally:
            cleanup()


# ── find_parallel (now an alias for find) ───────────────────────────


class TestFindParallel:
    """find_parallel() is a deprecated alias for find().

    The anchor-split implementation was removed. It measured 0.99-1.01x of
    find() on real 2.6.5, never once executed its parallel path (anchor_split
    accepts only the object's shard count, which is 1 on a single-target
    pool), would have called dfs_readdirplus_at on one shared handle from
    several threads if it ever had, and discarded worker exceptions so a
    failed partition returned a partial listing that looked complete.

    These tests pin the replacement contract: identical results to find(), and
    no anchor-split machinery touched.
    """

    def test_it_returns_exactly_what_find_returns(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            entries = [
                _make_mock_dir_entry("a.txt", size=10),
                _make_mock_dir_entry("b.txt", size=20),
            ]
            mock_B.dfs_lookup_path.return_value = (MagicMock(), TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = entries

            expected = fs.find("/mydir")
            result = fs.find_parallel("/mydir")

            assert result == expected
            assert result == ["/mydir/a.txt", "/mydir/b.txt"]
        finally:
            cleanup()

    def test_num_workers_is_accepted_and_ignored(self):
        """The parameter stays for compatibility; it must not change results."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.return_value = (MagicMock(), TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = [_make_mock_dir_entry("a.txt", size=1)]

            one = fs.find_parallel("/mydir", num_workers=1)
            many = fs.find_parallel("/mydir", num_workers=64)

            assert one == many == ["/mydir/a.txt"]
        finally:
            cleanup()

    def test_it_does_not_touch_the_anchor_split_api(self):
        """Proves the machinery is gone rather than merely unused by default."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.return_value = (MagicMock(), TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = [_make_mock_dir_entry("a.txt", size=1)]

            fs.find_parallel("/mydir", num_workers=8)

            assert mock_B.dfs_obj_anchor_split_optimal.call_count == 0
            assert mock_B.dfs_obj_anchor_split.call_count == 0
            assert mock_B.dfs_readdirplus_at.call_count == 0
        finally:
            cleanup()

    def test_detail_and_maxdepth_are_forwarded(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_B.dfs_lookup_path.return_value = (MagicMock(), TestFind._dir_stat())
            mock_B.dfs_readdirplus.return_value = [_make_mock_dir_entry("a.txt", size=7)]

            detailed = fs.find_parallel("/mydir", detail=True)

            assert isinstance(detailed, dict)
            assert detailed["/mydir/a.txt"]["size"] == 7
        finally:
            cleanup()


# ── async new methods ───────────────────────────────────────────────


class TestAsyncNewMethodsAll:
    """Additional async wrappers for new methods."""

    @patch("daos_fsspec.async_core.DFSFileSystemPool")
    def _make_fs(self, mock_pool_cls, **kwargs):
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_fs = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_fs)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.__repr__ = lambda s: (
            "DFSFileSystemPool(pool='test-pool', container='test-cont', size=4)"
        )
        fs = DFSAsyncFileSystem(pool="test-pool", container="test-cont", **kwargs)
        return fs, mock_fs, mock_pool

    @pytest.mark.asyncio
    async def test_chown(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._chown("/file.txt", 1000, 1000)
        mock_fs.chown.assert_called_once_with("/file.txt", 1000, 1000, follow_links=True)

    @pytest.mark.asyncio
    async def test_access(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.access.return_value = True
        result = await fs._access("/file.txt", mode=0)
        assert result is True
        mock_fs.access.assert_called_once_with("/file.txt", mode=0)

    @pytest.mark.asyncio
    async def test_mount_attr(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.mount_attr.return_value = MagicMock(chunk_size=1048576)
        result = await fs.mount_attr()
        assert result.chunk_size == 1048576
        mock_fs.mount_attr.assert_called_once()

    @pytest.mark.asyncio
    async def test_obj_info(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.obj_info.return_value = MagicMock(oclass_id=7, chunk_size=4194304)
        result = await fs._obj_info("/file.txt")
        assert result.oclass_id == 7
        mock_fs.obj_info.assert_called_once_with("/file.txt")

    @pytest.mark.asyncio
    async def test_truncate(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._truncate("/file.txt", size=50)
        mock_fs.truncate.assert_called_once_with("/file.txt", size=50)

    @pytest.mark.asyncio
    async def test_find_parallel(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.find_parallel.return_value = ["/a.txt", "/b.txt"]
        result = await fs._find_parallel("/dir")
        assert result == ["/a.txt", "/b.txt"]
        mock_fs.find_parallel.assert_called_once_with(
            "/dir", maxdepth=None, withdirs=False, detail=False, num_workers=4
        )


# ── async init passes new params ────────────────────────────────────


class TestAsyncInitNewParams:
    """AsyncFileSystem passes new params to pool."""

    def test_async_passes_object_class_and_chunk(self):
        with patch("daos_fsspec.async_core.DFSFileSystemPool") as mock_pool_cls:
            mock_pool_cls.return_value = MagicMock()
            DFSAsyncFileSystem(pool="p", container="c", object_class=7, chunk_size=4194304)
            call_kwargs = mock_pool_cls.call_args[1]
            assert call_kwargs["object_class"] == 7
            assert call_kwargs["chunk_size"] == 4194304


# ── DFSFile mock-based unit tests ──────────────────────────────────────


def _make_mock_fs_for_dfs_file(readonly=False):
    """Create a mock DFSFileSystem for DFSFile testing.

    Patches both daos_fsspec.core.B and daos_fsspec.dfs_file.B so that
    DFSFile constructor calls go through the mock (not real libdaos.so).
    """
    import gc

    gc.collect()

    mock_B = MagicMock()
    mock_B.DAOS_PC_RO = 1
    mock_B.DAOS_PC_RW = 2
    mock_B.DAOS_COO_RO = 1
    mock_B.DAOS_COO_RW = 2
    mock_B._CHUNK_SIZE = 16 * 1024 * 1024
    mock_B._HAS_DFS_CONNECT = False
    mock_B._ensure_libs = MagicMock()

    mock_root_obj = MagicMock(name="root_obj")
    mock_root_stat = MagicMock(name="root_stat")
    mock_dfs_handle = MagicMock(name="dfs_handle")
    mock_poh = MagicMock(name="poh")
    mock_coh = MagicMock(name="coh")

    mock_B.daos_init = MagicMock()
    mock_B.pool_connect = MagicMock(return_value=mock_poh)
    mock_B.cont_open = MagicMock(return_value=mock_coh)
    mock_B.dfs_mount_fs = MagicMock(return_value=mock_dfs_handle)
    mock_B.dfs_lookup_path = MagicMock(return_value=(mock_root_obj, mock_root_stat))
    mock_B.dfs_lookup_rel_path = MagicMock(return_value=(mock_root_obj, mock_root_stat))
    mock_B.dfs_release = MagicMock()
    mock_B.dfs_umount = MagicMock()
    mock_B.cont_close = MagicMock()
    mock_B.pool_disconnect = MagicMock()
    mock_B.dfs_open_file = MagicMock(return_value=MagicMock(name="file_obj"))
    mock_B.dfs_read = MagicMock(return_value=b"")
    mock_B.dfs_write = MagicMock()
    mock_B.dfs_close = MagicMock()

    from daos_fsspec._bindings import (
        DER_HG as _DER_HG,
    )
    from daos_fsspec._bindings import (
        DER_NO_HDL as _DER_NO_HDL,
    )
    from daos_fsspec._bindings import (
        DER_TIMEDOUT as _DER_TIMEDOUT,
    )
    from daos_fsspec._bindings import (
        DER_UNREACH as _DER_UNREACH,
    )
    from daos_fsspec._bindings import (
        RETRYABLE_ERRORS as _RETRYABLE_ERRORS,
    )

    class _MockDaosError(Exception):
        def __init__(self, rc=0, message=""):
            self.rc = rc
            super().__init__(message)

        @property
        def is_retryable(self):
            return self.rc in _RETRYABLE_ERRORS

    mock_B.DaosError = _MockDaosError
    mock_B.DER_UNREACH = _DER_UNREACH
    mock_B.DER_HG = _DER_HG
    mock_B.DER_NO_HDL = _DER_NO_HDL
    mock_B.DER_TIMEDOUT = _DER_TIMEDOUT
    mock_B.RETRYABLE_ERRORS = _RETRYABLE_ERRORS

    # Patch every module that imports the bindings; see conftest.patch_bindings.
    patcher = patch_bindings(mock_B)
    patcher.__enter__()

    from daos_fsspec.core import DFSFileSystem

    fs = DFSFileSystem(pool="test-pool", container="test-cont", readonly=readonly)
    mock_B.dfs_lookup_path.reset_mock()

    def cleanup():
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass
        patcher.__exit__(None, None, None)

    return fs, mock_B, cleanup


class TestDFSFileClose:
    """DFSFile.close() behavior."""

    def test_close_idempotent(self):
        """Double close should be safe."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            f.close()
            assert f._dfs_closed
            # Second close should be a no-op
            f.close()
            assert f._dfs_closed
        finally:
            cleanup()

    def test_close_releases_parent_handle(self):
        """close() releases parent handle if not root."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            # Use a deep path so _resolve_path returns a non-root parent
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/subdir/test.txt", "rb")
            parent = f._parent_obj
            f.close()

            # Which handle, not just that something was released.
            # assert_called() could not see the bug this covers; releasing
            # the *root* because the identity check compared Python wrappers
            # rather than addresses (tests/test_fault_injection.py pins that
            # directly).
            released = [c.args[0] for c in mock_B.dfs_release.call_args_list if c.args]
            assert any(r is parent for r in released), (
                f"the parent handle was not released; released {released}"
            )
            assert not any(r is fs._root for r in released), (
                "released the root handle, which must stay open for the mount"
            )
        finally:
            cleanup()

    def test_close_flush_error_still_releases(self):
        """close() releases DFS handle even if flush() raises."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "wb")

            # Patch super().close() to raise: the finally block still runs
            # and releases handles. The exception propagates after cleanup.
            with patch(
                "daos_fsspec.dfs_file.AbstractBufferedFile.close",
                side_effect=RuntimeError("flush error"),
            ):
                with pytest.raises(RuntimeError, match="flush error"):
                    f.close()

            # DFS handles were released despite the error
            assert f._dfs_closed
            assert f._dfs_obj is None
        finally:
            cleanup()


class TestDFSFileFetchRange:
    """DFSFile._fetch_range() behavior."""

    def test_fetch_range_basic(self):
        """_fetch_range returns data from dfs_read."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_read.return_value = b"hello world"

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            try:
                result = f._fetch_range(0, 11)
                assert result == b"hello world"
            finally:
                f.close()
        finally:
            cleanup()

    def test_fetch_range_partial(self):
        """_fetch_range with partial range."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_read.return_value = b"lo wo"

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            try:
                result = f._fetch_range(5, 10)
                assert result == b"lo wo"
            finally:
                f.close()
        finally:
            cleanup()

    def test_fetch_range_empty(self):
        """_fetch_range with empty range."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            try:
                result = f._fetch_range(5, 5)
                assert result == b""
            finally:
                f.close()
        finally:
            cleanup()

    def test_fetch_range_error(self):
        """_fetch_range propagates errors from dfs_read."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_read.side_effect = RuntimeError("I/O error")

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            try:
                with pytest.raises(RuntimeError, match="I/O error"):
                    f._fetch_range(0, 10)
            finally:
                f.close()
        finally:
            cleanup()


class TestDFSFileUploadChunk:
    """DFSFile._upload_chunk() behavior."""

    def test_upload_chunk_basic(self):
        """_upload_chunk writes buffered data to dfs_write."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=0, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "wb")
            try:
                f.buffer = MagicMock()
                f.buffer.getvalue.return_value = b"test data"
                f.offset = 0

                result = f._upload_chunk(final=True)
                assert result is True
                mock_B.dfs_write.assert_called_once()
            finally:
                f.close()
        finally:
            cleanup()

    def test_upload_chunk_empty_buffer(self):
        """_upload_chunk with empty buffer returns True without writing."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=0, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "wb")
            try:
                f.buffer = MagicMock()
                f.buffer.getvalue.return_value = b""

                result = f._upload_chunk()
                assert result is True
                mock_B.dfs_write.assert_not_called()
            finally:
                f.close()
        finally:
            cleanup()

    def test_upload_chunk_none_buffer(self):
        """_upload_chunk with None buffer returns True."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=0, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "wb")
            try:
                f.buffer = None
                result = f._upload_chunk()
                assert result is True
            finally:
                # Can't call f.close() with buffer=None: skip it
                f._dfs_closed = True
                f._dfs_obj = None
                f._parent_obj = None
        finally:
            cleanup()


class TestDFSFileDel:
    """DFSFile.__del__ safety net."""

    def test_del_when_closed(self):
        """__del__ does nothing when file is already closed."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            f.close()
            assert f._dfs_closed
            # __del__ should be a no-op
            f.__del__()
            assert f._dfs_closed
        finally:
            cleanup()

    def test_del_when_fs_closed(self):
        """__del__ does nothing when parent filesystem is closed."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            # Close parent filesystem
            fs.close()
            assert fs._closed
            # __del__ should not try to release handles (parent already closed)
            f.__del__()
            assert f._dfs_closed
        finally:
            cleanup()

    def test_del_when_not_closed(self):
        """__del__ releases handles when file is not closed."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            assert not f._dfs_closed
            # __del__ should release handles and set _dfs_closed
            f.__del__()
            assert f._dfs_closed
            # Should have called dfs_close
            mock_B.dfs_close.assert_called()
        finally:
            cleanup()

    def test_del_handles_module_none(self):
        """__del__ handles None _bindings module gracefully."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            # Simulate interpreter shutdown by removing dfs_close from B
            import daos_fsspec.dfs_file as df_module

            original_B = df_module.B
            try:
                df_module.B = None
                f.__del__()
            finally:
                df_module.B = original_B
        finally:
            cleanup()


class TestDFSFileRepr:
    """DFSFile.__repr__."""

    def test_repr(self):
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            try:
                r = repr(f)
                assert "DFSFile" in r
                assert "test.txt" in r
            finally:
                f.close()
        finally:
            cleanup()


class TestDFSFileModes:
    """DFSFile mode handling."""

    def test_read_mode(self):
        """Read mode opens file for reading."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "rb")
            try:
                assert f.size == 100
                assert not f._dfs_closed
            finally:
                f.close()
        finally:
            cleanup()

    def test_write_mode(self):
        """Write mode creates/truncates file."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=0, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_open_file.return_value = MagicMock(name="file_obj")

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "wb")
            try:
                assert f._dfs_obj is not None
                assert not f._dfs_closed
            finally:
                f.close()
        finally:
            cleanup()

    def test_append_mode_existing_file(self):
        """Append mode on existing file seeks to end."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=50, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "ab")
            try:
                assert f._dfs_obj is not None
                assert not f._dfs_closed
            finally:
                f.close()
        finally:
            cleanup()

    def test_append_mode_new_file(self):
        """Append mode creates file if it doesn't exist."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=0, mode=0o100644)
            mock_B.dfs_lookup_rel_path.side_effect = FileNotFoundError("not found")
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_open_file.return_value = MagicMock(name="file_obj")

            from daos_fsspec.dfs_file import DFSFile

            f = DFSFile(fs, "/test.txt", "ab")
            try:
                assert f._dfs_obj is not None
                assert not f._dfs_closed
            finally:
                f.close()
        finally:
            cleanup()

    def test_unsupported_mode(self):
        """Unsupported mode raises ValueError."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.return_value = (mock_parent, mock_stat)
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            with pytest.raises(ValueError, match="Unsupported mode"):
                DFSFile(fs, "/test.txt", "xb")
        finally:
            cleanup()


class TestDFSFileReadModeError:
    """DFSFile read mode error when file doesn't exist."""

    def test_read_mode_file_not_found(self):
        """Read mode raises FileNotFoundError if file doesn't exist."""
        fs, mock_B, cleanup = _make_mock_fs_for_dfs_file()
        try:
            mock_parent = MagicMock(name="parent")
            mock_stat = SimpleNamespace(size=100, mode=0o100644)
            mock_B.dfs_lookup_rel_path.side_effect = FileNotFoundError("not found")
            mock_B.dfs_lookup_path.return_value = (mock_parent, mock_stat)

            from daos_fsspec.dfs_file import DFSFile

            with pytest.raises(FileNotFoundError):
                DFSFile(fs, "/test.txt", "rb")
        finally:
            cleanup()


# ── Guard tests: _check_closed + _readonly on write operations ───────────


class TestRmGuards:
    """rm() enforces closed/readonly guards."""

    def test_rm_closed_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.rm("/file")
        finally:
            cleanup()

    def test_rm_readonly_raises(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.rm("/file")
        finally:
            cleanup()

    def test_rm_recursive_readonly_raises(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.rm("/dir", recursive=True)
        finally:
            cleanup()

    def test_rm_root_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(ValueError, match="root"):
                fs.rm("/", recursive=True)
        finally:
            cleanup()

    def test_rm_maxdepth_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(NotImplementedError, match="maxdepth"):
                fs.rm("/dir", recursive=True, maxdepth=1)
        finally:
            cleanup()


class TestReadClosedGuards:
    """Read-side operations fail fast after close()."""

    def test_read_methods_closed_raise(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            for method, args in [
                (fs.ls, ("/",)),
                (fs.info, ("/",)),
                (fs.exists, ("/",)),
                (fs.created, ("/",)),
                (fs.readlink, ("/link",)),
                (fs.islink, ("/link",)),
            ]:
                with pytest.raises(RuntimeError, match="closed"):
                    method(*args)

            with pytest.raises(RuntimeError, match="closed"):
                list(fs.walk("/"))
        finally:
            cleanup()


class TestRmdirGuards:
    """rmdir() enforces closed/readonly guards."""

    def test_rmdir_closed_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.rmdir("/dir")
        finally:
            cleanup()

    def test_rmdir_readonly_raises(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.rmdir("/dir")
        finally:
            cleanup()

    def test_rmdir_root_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(ValueError, match="root"):
                fs.rmdir("/")
        finally:
            cleanup()


class TestMvGuards:
    """mv() enforces closed/readonly guards and retries."""

    def test_mv_closed_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            fs.close()
            with pytest.raises(RuntimeError, match="closed"):
                fs.mv("/a", "/b")
        finally:
            cleanup()

    def test_mv_readonly_raises(self):
        fs, mock_B, cleanup = _make_mock_fs(readonly=True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                fs.mv("/a", "/b")
        finally:
            cleanup()

    def test_mv_root_raises(self):
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            with pytest.raises(ValueError, match="root"):
                fs.mv("/", "/dst")
            with pytest.raises(ValueError, match="root"):
                fs.mv("/src", "/")
        finally:
            cleanup()

    def test_mv_retries_on_transient_error(self):
        """mv retries on transient DAOS errors (same as rm/rmdir)."""
        fs, mock_B, cleanup = _make_mock_fs()
        try:
            mock_parent = MagicMock(name="parent_obj")
            mock_B.dfs_lookup_path.return_value = (mock_parent, MagicMock())
            mock_B.dfs_move.side_effect = [
                mock_B.DaosError(rc=mock_B.DER_TIMEDOUT, message="timeout"),
                None,  # succeeds on retry
            ]
            fs.mv("/src", "/dst")
            assert mock_B.dfs_move.call_count == 2
        finally:
            cleanup()
