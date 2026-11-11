# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for concurrency."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import patch_bindings

# How long each faked DAOS call "takes". Long enough that serialization is
# unambiguous, short enough to keep the suite fast.
_OP_DELAY = 0.05
_THREADS = 8


@pytest.fixture
def mock_fs():
    """A DFSFileSystem whose libdfs calls are replaced by a fixed sleep."""

    class _FakeDaosError(Exception):
        is_retryable = False

    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    # A real class, not Exception: _is_retryable() does an isinstance check
    # against it, and matching every exception would send failures down the
    # retry path and mask them.
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.pool_connect.return_value = MagicMock(name="poh")
    mock_B.cont_open.return_value = MagicMock(name="coh")
    mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="root"),
        SimpleNamespace(size=0, mode=0o040755),
    )

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="test-pool", container="test-cont")
        # _connect() looks up the root handle; don't let that count towards
        # the per-test call assertions.
        mock_B.dfs_lookup_path.reset_mock()
        yield fs, mock_B
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass


def _run_concurrently(fn, n=_THREADS):
    """Run fn in n threads; return wall-clock seconds and any error raised."""
    errors = []
    ready = threading.Barrier(n, timeout=10)

    def worker():
        try:
            ready.wait()
            fn()
        except Exception as exc:  # noqa: BLE001 - surfaced via the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.monotonic() - start
    return elapsed, errors


def test_reads_overlap(mock_fs):
    """The headline regression test.

    Serialized, 8 x 50ms is >= 400ms. Overlapped it is ~50ms. The threshold
    sits well below the serial figure so this stays meaningful on a loaded
    CI box, but well above the concurrent one.
    """
    fs, mock_B = mock_fs

    def slow_read(*args, **kwargs):
        time.sleep(_OP_DELAY)
        return b"x" * 16

    mock_B.dfs_read.side_effect = slow_read
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="obj"),
        SimpleNamespace(size=16, mode=0o100644),
    )

    elapsed, errors = _run_concurrently(lambda: fs.cat_file("/some/file"))

    assert not errors, f"concurrent reads raised: {errors}"
    # Guard against a vacuous pass: if the mock is wired to a function the
    # code never calls, nothing sleeps and any timing assertion succeeds.
    assert mock_B.dfs_read.call_count == _THREADS, (
        f"the slow path never ran ({mock_B.dfs_read.call_count} calls): "
        f"this test would pass for the wrong reason"
    )
    serial = _OP_DELAY * _THREADS
    assert elapsed < serial / 2, (
        f"reads serialized: {elapsed:.3f}s for {_THREADS} x {_OP_DELAY}s "
        f"(serial would be ~{serial:.3f}s)"
    )


def test_stats_overlap(mock_fs):
    """Metadata ops take the same shared lock as I/O."""
    fs, mock_B = mock_fs

    stat = SimpleNamespace(
        size=42,
        mode=0o100644,
        type_name="file",
        mtime=0,
        atime=0,
        uid=0,
        gid=0,
        is_dir=False,
    )

    def slow_lookup(*args, **kwargs):
        time.sleep(_OP_DELAY)
        return MagicMock(name="obj"), stat

    mock_B.dfs_lookup_path.side_effect = slow_lookup

    elapsed, errors = _run_concurrently(lambda: fs.info("/some/file"))

    assert not errors, f"concurrent stats raised: {errors}"
    assert mock_B.dfs_lookup_path.call_count == _THREADS, "the slow path never ran"
    assert elapsed < (_OP_DELAY * _THREADS) / 2, f"stats serialized: {elapsed:.3f}s"


def test_close_waits_for_in_flight_operations(mock_fs):
    """The one invariant the lock still has to enforce: teardown must not
    release handles out from under a call already inside libdfs."""
    fs, mock_B = mock_fs

    in_flight = threading.Event()
    finished = []

    def slow_read(*args, **kwargs):
        in_flight.set()
        time.sleep(0.3)
        finished.append("read")
        return b"data"

    mock_B.dfs_read.side_effect = slow_read
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="obj"),
        SimpleNamespace(size=4, mode=0o100644),
    )

    t = threading.Thread(target=lambda: fs.cat_file("/f"))
    t.start()
    assert in_flight.wait(timeout=5)

    fs.close()
    # close() must not have returned until the read left libdfs.
    assert finished == ["read"], "close() tore down handles during a live call"
    t.join(timeout=5)


_CLOSE_ORDERING_SCENARIO = """
import sys, threading, time
sys.path.insert(0, {tests!r})
sys.path.insert(0, {root!r})
from types import SimpleNamespace
from unittest.mock import MagicMock
from conftest import patch_bindings

class E(Exception):
    is_retryable = False

mb = MagicMock()
mb._HAS_DFS_CONNECT = False
mb.DaosError = E
mb.RETRYABLE_ERRORS = ()
mb.dfs_lookup_path.return_value = (MagicMock(), SimpleNamespace(size=0, mode=0o040755))

with patch_bindings(mb):
    from daos_fsspec.core import DFSFileSystem
    fs = DFSFileSystem(pool="p", container="c")
    ex = fs._executor
    n_workers = fs._CAT_RANGES_MAX_WORKERS

    holding = threading.Event()
    release = threading.Event()
    started = []
    counter_lock = threading.Lock()

    def hold_a_worker():
        # Occupies a worker *inside* the shared lock, which is what a
        # cat_ranges task looks like mid-read.
        with fs._lock.read():
            with counter_lock:
                started.append(1)
                if len(started) == n_workers:
                    holding.set()
            release.wait(30)

    def needs_the_lock():
        with fs._lock.read():
            pass

    for _ in range(n_workers):
        ex.submit(hold_a_worker)
    if not holding.wait(10):
        print("SETUP-FAILED: workers never occupied")
        raise SystemExit(3)

    for _ in range(20):
        ex.submit(needs_the_lock)
    if ex._work_queue.qsize() == 0:
        print("SETUP-FAILED: nothing queued behind the holders")
        raise SystemExit(3)

    done = threading.Event()
    threading.Thread(target=lambda: (fs.close(), done.set()), daemon=True).start()
    time.sleep(0.3)      # let close() reach its drain/lock step
    release.set()        # holders let go

    if not done.wait(20):
        print("DEADLOCK: close() never returned")
        raise SystemExit(4)
    print("OK")
"""


def test_close_drains_its_executor_outside_the_lock():
    """close() must not drain its executor while holding the lock they need."""
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    program = _CLOSE_ORDERING_SCENARIO.format(
        tests=str(pathlib.Path(__file__).resolve().parent), root=root
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=root,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "close() never returned; it is draining its own executor while "
            "holding the exclusive lock those workers need"
        )

    assert "SETUP-FAILED" not in result.stdout, f"the scenario did not arm: {result.stdout}"
    assert "DEADLOCK" not in result.stdout, result.stdout
    assert result.returncode == 0, (
        f"scenario exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_concurrent_reads_and_close_do_not_corrupt_state(mock_fs):
    """Readers racing teardown must either complete or raise cleanly: never
    hang, never touch handles after they are released."""
    fs, mock_B = mock_fs

    def quick_read(*args, **kwargs):
        time.sleep(0.005)
        return b"z"

    mock_B.dfs_read.side_effect = quick_read
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="obj"),
        SimpleNamespace(size=1, mode=0o100644),
    )

    stop = threading.Event()
    unexpected = []

    def churn():
        while not stop.is_set():
            try:
                fs.cat_file("/f")
            except RuntimeError:
                return  # expected once closed
            except Exception as exc:  # noqa: BLE001
                unexpected.append(exc)
                return

    threads = [threading.Thread(target=churn) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.1)

    fs.close()
    stop.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "reader hung across teardown"

    assert not unexpected, f"unexpected errors during teardown race: {unexpected}"


def test_close_captures_or_rejects_racing_lazy_executor_creation(mock_fs):
    """No executor may be created after close has detached the old one."""
    fs, _ = mock_fs
    result = {}

    fs._executor_lock.acquire()

    def get_executor():
        try:
            result["executor"] = fs._executor
        except BaseException as exc:  # noqa: BLE001 - asserted below
            result["error"] = exc

    creator = threading.Thread(target=get_executor, daemon=True)
    creator.start()
    closer = threading.Thread(target=fs.close, daemon=True)
    closer.start()

    deadline = time.monotonic() + 10
    while fs._lock._writer is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert fs._lock._writer is not None, "close never reached executor detachment"

    fs._executor_lock.release()
    creator.join(10)
    closer.join(10)

    assert not creator.is_alive() and not closer.is_alive()
    assert fs._cat_ranges_executor is None
    if "executor" in result:
        assert result["executor"]._shutdown is True, "racing executor escaped close()"
    else:
        assert isinstance(result.get("error"), RuntimeError)
