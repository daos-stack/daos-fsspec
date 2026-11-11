# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for runtime lifecycle."""

from __future__ import annotations

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lifecycle_library(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.fail("a C compiler is required for the lifecycle ABI regression tests")
    output = tmp_path_factory.mktemp("lifecycle-native") / "lifecycle.so"
    subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(output),
            str(ROOT / "tests/native/lifecycle.c"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


_SETUP = """
import ctypes, errno, gc, os, threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from daos_fsspec import _bindings as B
from daos_fsspec.core import DFSFileSystem
native = ctypes.CDLL(__import__('sys').argv[1])
native.test_fail.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
native.test_fail.restype = None
native.test_calls.argtypes = [ctypes.c_int]
native.test_calls.restype = ctypes.c_int
for name in ('test_daos_refs', 'test_dfs_refs'):
    getattr(native, name).argtypes = []
    getattr(native, name).restype = ctypes.c_int
daos, dfs = MagicMock(), MagicMock()
daos.daos_init, daos.daos_fini = native.daos_init, native.daos_fini
dfs.dfs_init, dfs.dfs_fini = native.dfs_init, native.dfs_fini
def connect(*args):
    args[-1]._obj.value = 42
    return 0
dfs.dfs_connect.side_effect = connect
dfs.dfs_disconnect.return_value = 0
dfs.dfs_release.return_value = 0
B._setup_signatures(daos, dfs)
patch.object(B, '_ensure_libs', return_value=(daos, dfs)).start()
patch.object(B, 'dfs_lookup_path', return_value=(
    B.dfs_obj_p(123), SimpleNamespace(size=0, mode=0o040755))).start()
patch.object(B, 'dfs_lookup_rel_path', return_value=(
    B.dfs_obj_p(456), SimpleNamespace(size=4, mode=0o100644))).start()
patch.object(B, 'dfs_open_file', return_value=B.dfs_obj_p(456)).start()
def build():
    return DFSFileSystem(pool='test-pool', container='test-cont')
def raises(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception:
        return
    raise AssertionError('expected failure')
"""


def run_scenario(library, program, *, expected_returncode=0, script_path=None, env=None):
    env = {**os.environ, **(env or {}), "PYTHONPATH": str(ROOT)}
    code = _SETUP + textwrap.dedent(program)
    command = [sys.executable, "-c", code, str(library)]
    if script_path is not None:
        script_path.write_text(code)
        command = [sys.executable, str(script_path), str(library)]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        # A hung forked child can keep capture pipes open after its parent dies.
        # Supervise the entire isolated process group, not only the parent PID.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"lifecycle process group exceeded 15s; stdout:\n{stdout}\nstderr:\n{stderr}")
    result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    assert result.returncode == expected_returncode, (
        f"exit: {result.returncode}; stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.mark.parametrize("dfs_init", [False, True])
def test_references_are_balanced_once(lifecycle_library, dfs_init):
    run_scenario(
        lifecycle_library,
        f"""
        initialize = B._dfs_init if {dfs_init!r} else B.daos_init
        for _ in range(20): initialize()
        assert native.test_daos_refs() == {2 if dfs_init else 1}
        B.daos_fini()
        B.daos_fini()
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
        assert native.test_calls(4) == {2 if dfs_init else 1}
        raises(initialize)
    """,
    )


def test_connections_and_reconnects_do_not_grow_runtime_references(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        for _ in range(20):
            fs = build()
            fs._reconnect()
            fs.close()
        assert native.test_calls(1) == 2
        assert native.test_calls(2) == 1
        B.daos_fini()
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
    """,
    )


def test_external_consumer_retains_its_references(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        assert native.daos_init() == native.dfs_init() == 0
        fs = build()
        fs.close()
        B.daos_fini()
        assert native.test_daos_refs() == 2 and native.test_dfs_refs() == 1
        assert native.dfs_fini() == native.daos_fini() == 0
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
    """,
    )


@pytest.mark.parametrize("stage", [1, 2])
def test_initialization_failure_does_not_claim_failed_reference(lifecycle_library, stage):
    run_scenario(
        lifecycle_library,
        f"""
        native.test_fail({stage}, 5, 0)
        raises(B._dfs_init)
        assert not B._runtime.dfs
        native.test_fail({stage}, 0, 0)
        B._dfs_init()
        B.daos_fini()
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
    """,
    )


@pytest.mark.parametrize("stage", [3, 4])
@pytest.mark.parametrize("after", [0, 1])
def test_failed_fini_is_never_retried(lifecycle_library, stage, after):
    run_scenario(
        lifecycle_library,
        f"""
        B._dfs_init()
        native.test_fail({stage}, 5, {after})
        raises(B.daos_fini)
        calls = native.test_calls({stage})
        raises(B.daos_fini)
        raises(B.daos_init)
        assert native.test_calls({stage}) == calls
    """,
    )


def test_explicit_fini_requires_closed_filesystems(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        raises(B.daos_fini)
        assert native.test_calls(3) == native.test_calls(4) == 0
        fs.close()
        B.daos_fini()
        assert native.test_daos_refs() == 0
    """,
    )


@pytest.mark.parametrize("failure", ["release", "disconnect"])
def test_cleanup_failure_blocks_fini_and_destructive_retry(lifecycle_library, failure):
    run_scenario(
        lifecycle_library,
        f"""
        fs = build()
        function = dfs.dfs_{failure}
        function.return_value = 5
        raises(fs.close)
        calls = function.call_count
        raises(fs.close)
        raises(B.daos_fini)
        B._runtime.at_exit()
        assert function.call_count == calls
        assert native.test_calls(3) == native.test_calls(4) == 0
    """,
    )


def test_failed_constructor_preserves_primary_error_and_cleanup_failure(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        B.dfs_lookup_path.side_effect = ValueError('lookup failed')
        dfs.dfs_disconnect.return_value = 5
        try: build()
        except ValueError as exc: assert str(exc) == 'lookup failed'
        else: raise AssertionError('lost original error')
        gc.collect()
        raises(B.daos_fini)
        B._runtime.at_exit()
        assert dfs.dfs_disconnect.call_count == 1
        assert native.test_calls(3) == 0
    """,
    )


def test_file_release_precedes_mount_without_flushing_buffers(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        file = fs.open('/write', 'wb')
        file.write(b'not committed')
        def disconnect(*args):
            assert dfs.dfs_release.call_count == 2
            return 0
        dfs.dfs_disconnect.side_effect = disconnect
        B._runtime.finish(automatic=True)
        dfs.dfs_write.assert_not_called()
        assert file.closed and fs._closed
        try:
            file.close()
        except OSError as exc:
            assert 'discarded' in str(exc)
        else:
            raise AssertionError('discarded writes were not reported')
        file.__del__()
        fs.__del__()
        assert dfs.dfs_release.call_count == 2
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
    """,
    )


def test_gc_closes_file_then_mount_before_fini(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        file = fs.open('/read', 'rb')
        del file
        gc.collect()
        assert not fs._open_files
        del fs
        gc.collect()
        assert not B._runtime.mounts
        B.daos_fini()
        assert native.test_daos_refs() == 0
    """,
    )


def test_fork_child_never_waits_for_inherited_runtime_lock(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        ready, done = threading.Event(), threading.Event()
        def hold():
            with B._runtime.lock:
                ready.set()
                done.wait(5)
        thread = threading.Thread(target=hold)
        thread.start()
        assert ready.wait(5)
        pid = os.fork()
        if pid == 0:
            B.daos_fini()
            raises(B.daos_init)
            assert native.test_calls(3) == native.test_calls(4) == 0
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        done.set()
        thread.join(5)
        assert os.waitstatus_to_exitcode(status) == 0
        fs.close()
        B.daos_fini()
    """,
    )


@pytest.mark.parametrize("cached", [False, True])
def test_real_exit_balances_refs_after_live_owner_cleanup(lifecycle_library, tmp_path, cached):
    import json

    trace = tmp_path / "native-exit.jsonl"
    result = run_scenario(
        lifecycle_library,
        f"""
        if {cached!r}:
            from daos_fsspec import get_fs
            fs = get_fs('test-pool', 'test-cont')
        else:
            fs = build()
        file = fs.open('/read', 'rb')
        # Observe after our callback, including later Python finalizers.
        import atexit
        def after():
            assert fs._closed and file.closed
            raises(build)
        atexit.unregister(B._runtime.at_exit)
        atexit.register(after)
        atexit.register(B._runtime.at_exit)
    """,
        env=dict(os.environ, DAOS_FSSPEC_NATIVE_EXIT_TRACE=str(trace)),
    )
    assert "Exception ignored" not in result.stderr and "cleanup failed" not in result.stderr
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["daos_refs"] == rows[0]["dfs_refs"] == 0
    assert rows[0]["dfs_fini"] == 1 and rows[0]["fini"] == 2


def test_unreleased_collected_file_prevents_mount_teardown(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        file = fs.open('/read', 'rb')
        # Simulate GC declining native cleanup because the mount lock is busy.
        file._dfs_closed = True
        del file
        gc.collect()
        raises(fs.close)
        dfs.dfs_disconnect.assert_not_called()
        raises(B.daos_fini)
        assert native.test_calls(3) == 0
    """,
    )


def test_file_close_failure_is_not_retried_by_exit(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        file = fs.open('/read', 'rb')
        dfs.dfs_release.return_value = 5
        raises(file.close)
        calls = dfs.dfs_release.call_count
        raises(fs.close)
        B._runtime.at_exit()
        file.__del__()
        assert dfs.dfs_release.call_count == calls
        dfs.dfs_disconnect.assert_not_called()
        assert native.test_calls(3) == 0
    """,
    )


def test_concurrent_initializers_own_one_reference(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _: B._dfs_init(), range(64)))
        assert native.test_calls(1) == 2
        assert native.test_calls(2) == 1
        B.daos_fini()
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
    """,
    )


def test_file_close_during_parent_close_keeps_handles_visible(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        file = fs.open('/read', 'rb')
        entered, finish = threading.Event(), threading.Event()
        def drain(**kwargs):
            entered.set()
            assert finish.wait(5)
        fs._cat_ranges_executor = SimpleNamespace(shutdown=drain)
        thread = threading.Thread(target=fs.close)
        thread.start()
        assert entered.wait(5)
        file.close()
        assert file.closed and file._dfs_obj is not None
        finish.set()
        thread.join(5)
        assert not thread.is_alive()
        assert dfs.dfs_release.call_count == 2
        assert not fs._open_files and not B._runtime.mounts
        B.daos_fini()
    """,
    )


def test_connections_overlap_and_fini_refuses_incomplete_construction(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        from concurrent.futures import ThreadPoolExecutor
        ready, release = threading.Barrier(3), threading.Event()
        original = dfs.dfs_connect.side_effect
        def connect(*args):
            ready.wait(5)
            assert release.wait(5)
            return original(*args)
        dfs.dfs_connect.side_effect = connect
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(build) for _ in range(2)]
            ready.wait(5)
            raises(B.daos_fini)
            raises(B._runtime.finish, automatic=True)
            assert native.test_calls(3) == 0
            release.set()
            filesystems = [future.result() for future in futures]
        for fs in filesystems: fs.close()
        B.daos_fini()
        assert native.test_daos_refs() == native.test_dfs_refs() == 0
    """,
    )


def test_recycled_identity_cannot_remove_an_orphan_registration(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        import weakref
        B._dfs_init()
        class Old: pass
        old = Old()
        reference = weakref.ref(old)
        del old
        gc.collect()
        fs = object.__new__(DFSFileSystem)
        B._runtime.mounts[id(fs)] = reference
        raises(DFSFileSystem.__init__, fs, pool='test-pool', container='test-cont')
        fs._close_from_finalizer()
        assert B._runtime.mounts[id(fs)] is reference
        raises(B.daos_fini)
        assert native.test_calls(3) == 0
    """,
    )


def test_flush_error_remains_primary_when_native_release_also_fails(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        fs = build()
        file = fs.open('/write', 'wb')
        file.write(b'buffered')
        dfs.dfs_release.return_value = 5
        with patch.object(file, '_upload_chunk', side_effect=ValueError('flush failed')):
            try: file.close()
            except ValueError as exc: assert str(exc) == 'flush failed'
            else: raise AssertionError('primary flush failure lost')
        assert B._runtime.failure is not None
        raises(fs.close)
        assert dfs.dfs_release.call_count == 1
        raises(B.daos_fini)
        assert native.test_calls(3) == 0
    """,
    )


def test_no_initialization_from_late_interpreter_finalizers(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        import sys
        with patch.object(sys, 'is_finalizing', return_value=True):
            raises(B.daos_init)
            raises(build)
        assert native.test_calls(1) == native.test_calls(2) == 0
        assert not B._runtime.registered
    """,
    )


def test_reentrant_finalization_cannot_overtake_initialization(lifecycle_library):
    run_scenario(
        lifecycle_library,
        """
        def initialize():
            raises(B.daos_fini)
            raises(B.daos_init)
            return native.daos_init()
        daos.daos_init = initialize
        B.daos_init()
        assert native.test_calls(1) == 1
        assert not B._runtime.finished
        B.daos_fini()
        assert native.test_daos_refs() == 0
    """,
    )
