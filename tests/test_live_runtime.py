# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for live runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["closed", "cached", "open_reader", "lowlevel", "external"])
def test_native_runtime_exit(daos_pool, daos_container, mode):
    program = r"""
import atexit, os, sys, uuid
from daos_fsspec import DFSFileSystem, get_fs
from daos_fsspec import _bindings as B
pool, container, mode = sys.argv[1:]
daos, dfs = B._ensure_libs()
if mode == 'external':
    B._check(daos.daos_init(), 'external daos_init')
    B._check(dfs.dfs_init(), 'external dfs_init')
if mode == 'lowlevel':
    B._HAS_DFS_CONNECT = False
fs = get_fs(pool, container) if mode == 'cached' else DFSFileSystem(pool, container)
path = '/lifecycle-' + uuid.uuid4().hex
fs.pipe_file(path, b'checked payload')
assert fs.cat_file(path) == b'checked payload'
if mode == 'cached':
    assert fs.cat_ranges([path] * 8, [0] * 8, [None] * 8) == [b'checked payload'] * 8
reader = fs.open(path, 'rb') if mode == 'open_reader' else None
if reader is not None:
    assert reader.read() == b'checked payload'
fs.rm(path)
if mode in ('closed', 'lowlevel', 'external'):
    fs.close()
    B.daos_fini()
    assert B._runtime.finished
if mode == 'external':
    # The package released only its references. An independently initialized
    # consumer can still connect and use DFS, then release its own references.
    import ctypes
    mount = B.dfs_p()
    B._check(dfs.dfs_connect(pool.encode(), None, container.encode(), os.O_RDONLY,
                            None, ctypes.byref(mount)), 'external dfs_connect')
    root, st = B.dfs_lookup_path(mount, '/', os.O_RDONLY)
    assert st.is_dir
    B._check(dfs.dfs_release(root), 'external release')
    B._check(dfs.dfs_disconnect(mount), 'external disconnect')
    B._check(dfs.dfs_fini(), 'external dfs_fini')
    B._check(daos.daos_fini(), 'external daos_fini')
def observe_after_cleanup():
    assert fs._closed and B._runtime.finished
    if reader is not None:
        assert reader.closed
        reader.__del__()
    fs.__del__()
    print('AFTER_NATIVE_CLEANUP', flush=True)
atexit.unregister(B._runtime.at_exit)
atexit.register(observe_after_cleanup)
atexit.register(B._runtime.at_exit)
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program), daos_pool, daos_container, mode],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AFTER_NATIVE_CLEANUP" in result.stdout, result.stdout + result.stderr
    assert "Exception ignored" not in result.stderr and "cleanup failed" not in result.stderr
