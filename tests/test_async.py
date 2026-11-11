# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for async."""

from unittest.mock import MagicMock, call, patch

import pytest

from daos_fsspec.async_core import DFSAsyncFileSystem


@pytest.fixture(autouse=True)
def _inline_to_thread(monkeypatch):
    """Run asyncio.to_thread inline for these mock-only unit tests.

    The async filesystem itself uses to_thread in production, but these tests
    assert delegation and argument passing with MagicMock objects. Running the
    mocks in real worker threads is unnecessary and can be flaky on minimal CI
    runtimes.
    """

    async def _run_inline(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("daos_fsspec.async_core.asyncio.to_thread", _run_inline)


class TestAsyncFileSystemInit:
    """Test async filesystem initialization."""

    def test_protocol(self):
        assert DFSAsyncFileSystem.protocol == "dfs"

    def test_cachable_false(self):
        assert DFSAsyncFileSystem.cachable is False

    def test_has_required_methods(self):
        required = [
            "_ls",
            "_info",
            "_exists",
            "_cat_file",
            "_pipe_file",
            "_mkdir",
            "_makedirs",
            "_rm",
            "_rm_file",
            "_get_file",
            "_put_file",
            "_cp_file",
            "_mv",
            "_symlink",
            "_readlink",
            "_touch",
            "_walk",
            "_cat_ranges",
            "_find",
            "close",
            "health_check",
        ]
        for method in required:
            assert hasattr(DFSAsyncFileSystem, method), f"Missing method: {method}"


class TestAsyncFileSystemOps:
    """Async filesystem operations: mock-based, no DAOS.

    Patches DFSFileSystemPool so no real connections are created.
    Each test gets a mock pool with a mock connection context manager.
    """

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
    async def test_ls(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.ls.return_value = ["/a", "/b"]
        result = await fs._ls("/")
        assert result == ["/a", "/b"]
        mock_fs.ls.assert_called_once_with("/", detail=False)

    @pytest.mark.asyncio
    async def test_ls_detail(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.ls.return_value = [{"name": "/a", "type": "file"}]
        result = await fs._ls("/", detail=True)
        assert result == [{"name": "/a", "type": "file"}]
        mock_fs.ls.assert_called_once_with("/", detail=True)

    @pytest.mark.asyncio
    async def test_info(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.info.return_value = {"name": "/f", "size": 100, "type": "file"}
        result = await fs._info("/f")
        assert result["size"] == 100

    @pytest.mark.asyncio
    async def test_exists(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.exists.return_value = True
        assert await fs._exists("/f") is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.exists.return_value = False
        assert await fs._exists("/nope") is False

    @pytest.mark.asyncio
    async def test_cat_file(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.cat_file.return_value = b"hello"
        result = await fs._cat_file("/f")
        assert result == b"hello"

    @pytest.mark.asyncio
    async def test_cat_file_range(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.cat_file.return_value = b"el"
        result = await fs._cat_file("/f", start=1, end=3)
        assert result == b"el"
        mock_fs.cat_file.assert_called_once_with("/f", start=1, end=3)

    @pytest.mark.asyncio
    async def test_pipe_file(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._pipe_file("/f", b"data")
        mock_fs.pipe_file.assert_called_once_with("/f", b"data")

    @pytest.mark.asyncio
    async def test_mkdir(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._mkdir("/dir")
        mock_fs.mkdir.assert_called_once_with("/dir", create_parents=True)

    @pytest.mark.asyncio
    async def test_makedirs(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._makedirs("/dir/sub")
        mock_fs.makedirs.assert_called_once_with("/dir/sub", exist_ok=False)

    @pytest.mark.asyncio
    async def test_rm(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._rm("/f")
        mock_fs.rm.assert_called_once_with("/f", recursive=False, maxdepth=None)

    @pytest.mark.asyncio
    async def test_rm_drops_extra_kwargs(self):
        """_rm must NOT forward **kwargs, and that was untested.

        The sync rm() has a fixed signature (path, recursive, maxdepth), so
        forwarding fsspec's batch_size= raises TypeError. async_core carries a
        comment explaining this; reverting the code to forward **kwargs passed
        both rm tests, because neither passed an extra kwarg.
        test_rm_file_ignores_extra_kwargs already did this correctly for
        _rm_file.
        """
        fs, mock_fs, _ = self._make_fs()

        await fs._rm("/f", batch_size=128, on_error="omit")

        mock_fs.rm.assert_called_once_with("/f", recursive=False, maxdepth=None)

    @pytest.mark.asyncio
    async def test_rm_recursive(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._rm("/dir", recursive=True)
        mock_fs.rm.assert_called_once_with("/dir", recursive=True, maxdepth=None)

    @pytest.mark.asyncio
    async def test_rm_file_ignores_extra_kwargs(self):
        # fsspec's _rm_file passes extras like batch_size=; must not leak to
        # the sync rm() (which has a fixed signature) and raise TypeError.
        fs, mock_fs, _ = self._make_fs()
        await fs._rm_file("/f", batch_size=1)
        mock_fs.rm.assert_called_once_with("/f")

    @pytest.mark.asyncio
    async def test_get_file(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._get_file("/remote", "/local")
        mock_fs.get_file.assert_called_once_with("/remote", "/local")

    @pytest.mark.asyncio
    async def test_put_file(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._put_file("/local", "/remote")
        mock_fs.put_file.assert_called_once_with("/local", "/remote")

    @pytest.mark.asyncio
    async def test_cp_file(self):
        fs, mock_fs, _ = self._make_fs()
        src = MagicMock()
        src.read.side_effect = [b"pay", b"load", b""]
        dst = MagicMock()
        mock_fs.open.side_effect = [
            MagicMock(
                __enter__=MagicMock(return_value=src), __exit__=MagicMock(return_value=False)
            ),
            MagicMock(
                __enter__=MagicMock(return_value=dst), __exit__=MagicMock(return_value=False)
            ),
        ]

        await fs._cp_file("/src", "/dst")

        mock_fs.open.assert_any_call("/src", "rb")
        mock_fs.open.assert_any_call("/dst", "wb")
        assert dst.write.call_args_list == [call(b"pay"), call(b"load")]

    @pytest.mark.asyncio
    async def test_mv(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._mv("/a", "/b")
        mock_fs.mv.assert_called_once_with("/a", "/b")

    @pytest.mark.asyncio
    async def test_symlink(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._symlink("/link", "/target")
        mock_fs.symlink.assert_called_once_with("/link", "/target")

    @pytest.mark.asyncio
    async def test_readlink(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.readlink.return_value = "/target"
        result = await fs._readlink("/link")
        assert result == "/target"
        mock_fs.readlink.assert_called_once_with("/link")

    @pytest.mark.asyncio
    async def test_touch(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._touch("/file")
        mock_fs.touch.assert_called_once_with("/file", truncate=True)

    @pytest.mark.asyncio
    async def test_walk(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.walk.return_value = iter(
            [
                ("/dir", ["sub"], ["file.txt"]),
                ("/dir/sub", [], ["inner.txt"]),
            ]
        )
        results = []
        async for item in fs._walk("/dir"):
            results.append(item)
        assert len(results) == 2
        assert results[0][0] == "/dir"

    @pytest.mark.asyncio
    async def test_cat_ranges(self):
        fs, mock_fs, _ = self._make_fs()
        calls = []

        def cat_file(path, **kwargs):
            calls.append((path, kwargs))
            return {"/f1": b"a", "/f2": b"b"}[path]

        mock_fs.cat_file = cat_file
        result = await fs._cat_ranges(["/f1", "/f2"], [0, 0], [5, 5])
        assert result == [b"a", b"b"]
        assert calls == [
            ("/f1", {"start": 0, "end": 5}),
            ("/f2", {"start": 0, "end": 5}),
        ]

    @pytest.mark.asyncio
    async def test_cat_ranges_scalar_broadcast(self):
        fs, mock_fs, _ = self._make_fs()
        calls = []

        def cat_file(path, **kwargs):
            calls.append((path, kwargs))
            return {"/a": b"x", "/b": b"y"}[path]

        mock_fs.cat_file = cat_file
        result = await fs._cat_ranges(["/a", "/b"], 0, 10)
        assert result == [b"x", b"y"]
        assert calls == [
            ("/a", {"start": 0, "end": 10}),
            ("/b", {"start": 0, "end": 10}),
        ]

    @pytest.mark.asyncio
    async def test_cat_ranges_on_error_return(self):
        fs, mock_fs, _ = self._make_fs()
        err = OSError("bad")

        def cat_file(path, **kwargs):
            if path == "/bad":
                raise err
            return b"ok"

        mock_fs.cat_file = cat_file
        result = await fs._cat_ranges(["/ok", "/bad"], [0, 0], [5, 5], on_error="return")
        assert result[0] == b"ok"
        assert result[1] is err

    @pytest.mark.asyncio
    async def test_cat_ranges_on_error_raise(self):
        fs, mock_fs, _ = self._make_fs()

        def cat_file(path, **kwargs):
            if path == "/bad":
                raise OSError("bad")
            return b"ok"

        mock_fs.cat_file = cat_file
        with pytest.raises(OSError, match="bad"):
            await fs._cat_ranges(["/ok", "/bad"], [0, 0], [5, 5], on_error="raise")

    @pytest.mark.asyncio
    async def test_cat_ranges_validates_lengths(self):
        fs, _, _ = self._make_fs()
        with pytest.raises(ValueError, match="matching paths length"):
            await fs._cat_ranges(["/a", "/b"], [0], [5, 5])
        with pytest.raises(ValueError, match="matching paths length"):
            await fs._cat_ranges(["/a", "/b"], [0, 0], [5])

    @pytest.mark.asyncio
    async def test_cat_ranges_validates_on_error(self):
        fs, _, _ = self._make_fs()
        with pytest.raises(ValueError, match="on_error"):
            await fs._cat_ranges(["/a"], [0], [5], on_error="ignore")

    @pytest.mark.asyncio
    async def test_find(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.find.return_value = ["/dir/a.txt", "/dir/b.txt"]
        result = await fs._find("/dir")
        assert result == ["/dir/a.txt", "/dir/b.txt"]
        mock_fs.find.assert_called_once_with("/dir", maxdepth=None, withdirs=False, detail=False)

    @pytest.mark.asyncio
    async def test_close(self):
        fs, _, mock_pool = self._make_fs()
        await fs.close()
        mock_pool.close.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_health_check(self):
        fs, _, mock_pool = self._make_fs()
        mock_pool.health_check.return_value = {"connection_0": True, "connection_1": True}
        result = await fs.health_check()
        assert result == {"connection_0": True, "connection_1": True}

    def test_repr(self):
        fs, _, _ = self._make_fs()
        r = repr(fs)
        assert "DFSAsyncFileSystem" in r
        assert "test-pool" in r

    def test_pool_size_passed(self):
        """Custom size is passed to DFSFileSystemPool."""
        with patch("daos_fsspec.async_core.DFSFileSystemPool") as mock_pool_cls:
            mock_pool_cls.return_value = MagicMock()
            DFSAsyncFileSystem(pool="p", container="c", size=8)
            mock_pool_cls.assert_called_once()
            call_kwargs = mock_pool_cls.call_args[1]
            assert call_kwargs["size"] == 8

    def test_pool_readonly_passed(self):
        """Readonly flag is passed to DFSFileSystemPool."""
        with patch("daos_fsspec.async_core.DFSFileSystemPool") as mock_pool_cls:
            mock_pool_cls.return_value = MagicMock()
            DFSAsyncFileSystem(pool="p", container="c", readonly=True)
            call_kwargs = mock_pool_cls.call_args[1]
            assert call_kwargs["readonly"] is True

    def test_pool_object_class_passed(self):
        """object_class is passed to DFSFileSystemPool."""
        with patch("daos_fsspec.async_core.DFSFileSystemPool") as mock_pool_cls:
            mock_pool_cls.return_value = MagicMock()
            DFSAsyncFileSystem(pool="p", container="c", object_class=7, chunk_size=4194304)
            call_kwargs = mock_pool_cls.call_args[1]
            assert call_kwargs["object_class"] == 7
            assert call_kwargs["chunk_size"] == 4194304


class TestAsyncNewMethods:
    """Async wrappers for new methods (xattr, head, tail, du, checksum, etc.)."""

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
    async def test_head(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.head.return_value = b"first1024"
        result = await fs._head("/file.txt", size=1024)
        assert result == b"first1024"
        mock_fs.head.assert_called_once_with("/file.txt", size=1024)

    @pytest.mark.asyncio
    async def test_tail(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.tail.return_value = b"last1024"
        result = await fs._tail("/file.txt", size=1024)
        assert result == b"last1024"
        mock_fs.tail.assert_called_once_with("/file.txt", size=1024)

    @pytest.mark.asyncio
    async def test_getxattr(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.getxattr.return_value = b"abc123"
        result = await fs._getxattr("/file.txt", "user.md5")
        assert result == b"abc123"
        mock_fs.getxattr.assert_called_once_with("/file.txt", "user.md5")

    @pytest.mark.asyncio
    async def test_setxattr(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._setxattr("/file.txt", "user.md5", b"abc123")
        mock_fs.setxattr.assert_called_once_with("/file.txt", "user.md5", b"abc123", flags=0)

    @pytest.mark.asyncio
    async def test_setxattr_with_flags(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._setxattr("/file.txt", "user.new", b"v", flags=1)
        mock_fs.setxattr.assert_called_once_with("/file.txt", "user.new", b"v", flags=1)

    @pytest.mark.asyncio
    async def test_listxattr(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.listxattr.return_value = ["user.md5", "user.sha256"]
        result = await fs._listxattr("/file.txt")
        assert result == ["user.md5", "user.sha256"]
        mock_fs.listxattr.assert_called_once_with("/file.txt")

    @pytest.mark.asyncio
    async def test_removexattr(self):
        fs, mock_fs, _ = self._make_fs()
        await fs._removexattr("/file.txt", "user.md5")
        mock_fs.removexattr.assert_called_once_with("/file.txt", "user.md5")

    @pytest.mark.asyncio
    async def test_du(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.du.return_value = {"/dir/a.txt": 100, "/dir/b.txt": 200}
        result = await fs._du("/dir")
        assert result == {"/dir/a.txt": 100, "/dir/b.txt": 200}
        mock_fs.du.assert_called_once_with("/dir", total=False, maxdepth=None)

    @pytest.mark.asyncio
    async def test_du_total(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.du.return_value = 300
        result = await fs._du("/dir", total=True)
        assert result == 300
        mock_fs.du.assert_called_once_with("/dir", total=True, maxdepth=None)

    @pytest.mark.asyncio
    async def test_checksum(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.checksum.return_value = "33188:100:1700000000.000000"
        result = await fs._checksum("/file.txt")
        assert result == "33188:100:1700000000.000000"
        mock_fs.checksum.assert_called_once_with("/file.txt")

    @pytest.mark.asyncio
    async def test_created(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.created.return_value = 1700000000.0
        result = await fs._created("/file.txt")
        assert result == 1700000000.0
        mock_fs.created.assert_called_once_with("/file.txt")

    @pytest.mark.asyncio
    async def test_modified(self):
        fs, mock_fs, _ = self._make_fs()
        mock_fs.modified.return_value = 1700000000.0
        result = await fs._modified("/file.txt")
        assert result == 1700000000.0
        mock_fs.modified.assert_called_once_with("/file.txt")

    @pytest.mark.asyncio
    async def test_pool_space(self):
        fs, mock_fs, _ = self._make_fs()
        mock_space = MagicMock(scm_total=1024)
        mock_fs.pool_space.return_value = mock_space
        result = await fs.pool_space()
        assert result.scm_total == 1024
        mock_fs.pool_space.assert_called_once()


class TestRegisterImplementation:
    """Test that the dfs:// protocol is registered on import."""

    def test_dfs_protocol_registered(self):
        import fsspec

        cls = fsspec.get_filesystem_class("dfs")
        assert cls.__name__ == "DFSFileSystem"

    def test_dfs_protocol_from_import(self):
        """Protocol is registered on import, not just on entry point."""
        import importlib

        import daos_fsspec

        importlib.reload(daos_fsspec)
        import fsspec

        cls = fsspec.get_filesystem_class("dfs")
        assert cls is daos_fsspec.DFSFileSystem

    def test_dfs_protocol_clobber(self):
        """Re-registering must actually overwrite, not raise or no-op.

        This was a verbatim duplicate of test_dfs_protocol_registered: it
        never called register_implementation, so the thing its name describes
        went untested while the file appeared to cover it twice.
        """
        import fsspec

        from daos_fsspec import DFSFileSystem

        class _Stand_in(DFSFileSystem):
            pass

        try:
            fsspec.register_implementation("dfs", _Stand_in, clobber=True)
            assert fsspec.get_filesystem_class("dfs") is _Stand_in, (
                "clobber=True did not replace the registered class"
            )
        finally:
            fsspec.register_implementation("dfs", DFSFileSystem, clobber=True)

        assert fsspec.get_filesystem_class("dfs") is DFSFileSystem, (
            "failed to restore the real registration"
        )


class TestTypeStubs:
    """Test that the .pyi stub file exists and is syntactically valid."""

    def test_pyi_exists(self):
        import os

        stub_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "daos_fsspec",
            "_bindings.pyi",
        )
        assert os.path.exists(stub_path), f"Stub file not found: {stub_path}"

    def test_pyi_parseable(self):
        """The .pyi file should be valid Python syntax."""
        import ast
        import os

        stub_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "daos_fsspec",
            "_bindings.pyi",
        )
        with open(stub_path) as f:
            content = f.read()
        # Should not raise SyntaxError
        tree = ast.parse(content)
        # Should have class and function definitions
        names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "StatResult" in names
        assert "DirEntry" in names
        assert "DaosError" in names
        assert "daos_init" in names
        assert "dfs_read" in names
        assert "dfs_write" in names

    def test_pyi_covers_public_api(self):
        """Stub should cover all public functions from _bindings."""
        import ast
        import os

        stub_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "daos_fsspec",
            "_bindings.pyi",
        )
        with open(stub_path) as f:
            tree = ast.parse(f.read())

        stub_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        # Core functions that must be in the stub
        required = [
            "daos_init",
            "daos_fini",
            "pool_connect",
            "pool_disconnect",
            "cont_open",
            "cont_close",
            "cont_create",
            "cont_destroy",
            "pool_list_cont",
            "dfs_mount_fs",
            "dfs_umount",
            "dfs_lookup_path",
            "dfs_lookup_rel_path",
            "dfs_release",
            "dfs_stat",
            "dfs_setattr",
            "dfs_open_file",
            "dfs_read",
            "dfs_write",
            "dfs_close",
            "dfs_readdirplus",
            "dfs_mkdir",
            "dfs_remove",
            "dfs_move",
            "dfs_create_symlink",
            "dfs_read_symlink",
        ]
        for name in required:
            assert name in stub_names, f"Missing from stub: {name}"
