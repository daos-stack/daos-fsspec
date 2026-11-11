# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for pipelined reads."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import patch_bindings

from daos_fsspec import _bindings as B

CHUNK = B.CHUNK_SIZE


class _FakeDaosError(Exception):
    is_retryable = False


@pytest.fixture
def fs_and_mock():
    """A DFSFileSystem whose dfs_read is programmable."""
    mock_B = MagicMock()
    mock_B._HAS_DFS_CONNECT = False
    mock_B.DaosError = _FakeDaosError
    mock_B.RETRYABLE_ERRORS = ()
    mock_B.CHUNK_SIZE = CHUNK
    mock_B.pool_connect.return_value = MagicMock(name="poh")
    mock_B.cont_open.return_value = MagicMock(name="coh")
    mock_B.dfs_mount_fs.return_value = MagicMock(name="dfs")
    mock_B.dfs_lookup_path.return_value = (
        MagicMock(name="obj"),
        SimpleNamespace(size=0, mode=0o040755, type_name="directory", is_dir=True),
    )
    # Real 2-tuple: _resolve_path and DFSFile.__init__ both unpack this, and a
    # bare MagicMock makes them raise ValueError before the code under test
    # runs; a test would then pass or fail on the wrong thing entirely.
    mock_B.dfs_lookup_rel_path.return_value = (
        MagicMock(name="child"),
        SimpleNamespace(size=0, mode=0o100644, type_name="file", is_dir=False),
    )
    mock_B.dfs_open_file.return_value = MagicMock(name="fileobj")

    with patch_bindings(mock_B):
        from daos_fsspec.core import DFSFileSystem

        fs = DFSFileSystem(pool="p", container="c")
        yield fs, mock_B
        try:
            if not fs._closed:
                fs.close()
        except Exception:
            pass


def _backing(total: int, *, short_at: int | None = None, short_to: int = 1024):
    """A dfs_read implementation over a synthetic file of `total` bytes."""
    data = bytes((i % 251) for i in range(total))
    state = {"shorted": False}

    def dfs_read(_dfs, _obj, offset, size):
        if offset >= total:
            return b""
        avail = min(size, total - offset)
        if short_at is not None and offset == short_at and not state["shorted"]:
            state["shorted"] = True
            avail = min(avail, short_to)
        return data[offset : offset + avail]

    return data, dfs_read


class TestCorrectness:
    def test_small_read_takes_the_single_call_path(self, fs_and_mock):
        """Below the threshold there must be no thread hand-off at all."""
        fs, mock_B = fs_and_mock
        data, reader = _backing(1024)
        mock_B.dfs_read.side_effect = reader

        got = fs._read_pipelined(MagicMock(), 0, 1024)

        assert got == data
        assert mock_B.dfs_read.call_count == 1

    def test_large_read_is_split_into_chunks(self, fs_and_mock):
        """Above the threshold the read is issued as CHUNK_SIZE pieces."""
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        data, reader = _backing(total)
        mock_B.dfs_read.side_effect = reader

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert got == data, "pipelined read returned wrong bytes"
        assert mock_B.dfs_read.call_count == 6, (
            f"expected 6 chunk reads, got {mock_B.dfs_read.call_count}"
        )

    def test_reassembly_is_in_offset_order(self, fs_and_mock):
        """Chunks complete out of order; the result must still be in order.

        The backing data is position-dependent, so a swapped pair fails on
        content rather than on length.
        """
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        data, reader = _backing(total)

        # Make later chunks finish first.
        def staggered(_dfs, _obj, offset, size):
            time.sleep(0.05 * (1 - offset / total))
            return reader(_dfs, _obj, offset, size)

        mock_B.dfs_read.side_effect = staggered

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert got == data, "chunks were reassembled out of order"

    def test_offset_read_is_not_assumed_to_start_at_zero(self, fs_and_mock):
        """offset read is not assumed to start at zero."""
        fs, mock_B = fs_and_mock
        total = 12 * CHUNK
        data, reader = _backing(total)
        mock_B.dfs_read.side_effect = reader

        start, size = CHUNK + 7, 6 * CHUNK
        assert start + size < total, "range must not reach EOF for this test"
        got = fs._read_pipelined(MagicMock(), start, size)

        assert len(got) == size
        assert got == data[start : start + size]

    def test_non_chunk_multiple_size_is_exact(self, fs_and_mock):
        """The tail chunk must request only what remains, not a full chunk."""
        fs, mock_B = fs_and_mock
        total = 5 * CHUNK + 12345
        data, reader = _backing(total)
        mock_B.dfs_read.side_effect = reader

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert len(got) == total
        assert got == data


class TestShortReadContract:
    def test_a_short_read_mid_chunk_is_not_treated_as_eof(self, fs_and_mock):
        """The bug class this guards: short != EOF.

        dfs_read may return fewer bytes than asked for at a chunk or stripe
        boundary. A worker that stopped there would silently drop the rest of
        its range, and the result would be a plausible but truncated file.
        """
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        data, reader = _backing(total, short_at=CHUNK, short_to=4096)
        mock_B.dfs_read.side_effect = reader

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert len(got) == total, f"short read truncated the result to {len(got)} of {total} bytes"
        assert got == data

    def test_repeated_short_reads_still_complete(self, fs_and_mock):
        """Every call short by design; only an empty read ends a range."""
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        data = bytes((i % 251) for i in range(total))

        def always_short(_dfs, _obj, offset, size):
            if offset >= total:
                return b""
            avail = min(size, total - offset, 8192)
            return data[offset : offset + avail]

        mock_B.dfs_read.side_effect = always_short

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert got == data, "a sequence of short reads did not reassemble"

    def test_eof_inside_a_chunk_truncates_and_drops_later_chunks(self, fs_and_mock):
        """Genuine EOF: an empty read.

        The file is shorter than the requested range. The result must stop at
        the real end of data; never splice in later pieces across the hole,
        and never pad.
        """
        fs, mock_B = fs_and_mock
        real_total = 2 * CHUNK + 500
        data, reader = _backing(real_total)
        mock_B.dfs_read.side_effect = reader

        got = fs._read_pipelined(MagicMock(), 0, 8 * CHUNK)

        assert len(got) == real_total, f"expected {real_total} bytes, got {len(got)}"
        assert got == data

    def test_a_hole_is_never_spliced_over(self, fs_and_mock):
        """If a middle chunk ends early, later data must not be appended.

        Appending chunk 3 after a short chunk 1 would produce bytes at the
        wrong offsets; corruption that looks like valid data.
        """
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        data = bytes((i % 251) for i in range(total))

        def eof_in_second_chunk(_dfs, _obj, offset, size):

            if offset == CHUNK:
                return data[CHUNK : CHUNK + 100]
            if CHUNK < offset < 2 * CHUNK:
                return b""
            if offset >= 2 * CHUNK:
                return data[offset : offset + min(size, total - offset)]
            return data[offset : offset + min(size, total - offset)]

        mock_B.dfs_read.side_effect = eof_in_second_chunk

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert got == data[: CHUNK + 100], (
            "data past the EOF hole was spliced in; result length "
            f"{len(got)} vs expected {CHUNK + 100}"
        )

    def test_empty_first_chunk_returns_empty(self, fs_and_mock):
        fs, mock_B = fs_and_mock
        mock_B.dfs_read.side_effect = lambda *a: b""

        got = fs._read_pipelined(MagicMock(), 0, 6 * CHUNK)

        assert got == b""

    @pytest.mark.parametrize("first_fails", [False, True], ids=["early-eof", "read-error"])
    def test_cat_file_drains_every_worker_before_releasing_its_handle(
        self, fs_and_mock, first_fails
    ):
        """An early exit must not free ``obj`` under later chunk workers."""
        fs, mock_B = fs_and_mock
        total = 8 * CHUNK
        obj = MagicMock(name="shared-file-object")
        mock_B.dfs_lookup_path.return_value = (
            obj,
            SimpleNamespace(size=total, mode=0o100644, type_name="file", is_dir=False),
        )

        lock = threading.Lock()
        release_later = threading.Event()
        all_later_started = threading.Event()
        state = {"active": 0, "later_started": 0, "released_while_active": False}

        def read(_dfs, _obj, offset, size):
            assert _obj is obj
            if offset == 0:
                assert all_later_started.wait(5), "later workers never started"
                if first_fails:
                    raise OSError("injected first-chunk failure")
                return b""
            with lock:
                state["active"] += 1
                state["later_started"] += 1
                if state["later_started"] == 7:
                    all_later_started.set()
            try:
                assert release_later.wait(5), "test did not release later workers"
                return b"x" * size
            finally:
                with lock:
                    state["active"] -= 1

        def release(handle):
            if handle is obj:
                with lock:
                    state["released_while_active"] = state["active"] != 0

        mock_B.dfs_read.side_effect = read
        mock_B.dfs_release.side_effect = release
        outcome = {}

        def run():
            try:
                outcome["value"] = fs.cat_file("/short.bin")
            except BaseException as exc:  # noqa: BLE001 - asserted below
                outcome["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert all_later_started.wait(5), "pipeline did not occupy every worker"
        time.sleep(0.05)
        assert thread.is_alive(), "cat_file returned without draining active workers"
        assert not any(
            call.args and call.args[0] is obj for call in mock_B.dfs_release.call_args_list
        ), "object was released while later reads were still blocked"

        release_later.set()
        thread.join(timeout=5)
        assert not thread.is_alive(), "cat_file did not finish after workers drained"
        assert state["active"] == 0
        assert state["released_while_active"] is False
        if first_fails:
            assert isinstance(outcome.get("error"), OSError)
        else:
            assert outcome.get("value") == b""


class TestOverlap:
    def test_chunk_reads_actually_overlap(self, fs_and_mock):
        """The reads must run concurrently, not just be submitted."""
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        data = bytes((i % 251) for i in range(total))
        lock = threading.Lock()
        state = {"now": 0, "peak": 0}

        def tracked(_dfs, _obj, offset, size):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            try:
                time.sleep(0.1)
                return data[offset : offset + min(size, total - offset)]
            finally:
                with lock:
                    state["now"] -= 1

        mock_B.dfs_read.side_effect = tracked

        got = fs._read_pipelined(MagicMock(), 0, total)

        assert got == data
        assert state["peak"] > 1, (
            f"chunk reads never overlapped (peak concurrency {state['peak']}); "
            "the pipelined path is serialising"
        )

    def test_worker_count_is_bounded(self, fs_and_mock):
        """Concurrency must stay within the executor's cap."""
        fs, mock_B = fs_and_mock
        total = 32 * CHUNK
        lock = threading.Lock()
        state = {"now": 0, "peak": 0}

        def tracked(_dfs, _obj, offset, size):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            try:
                time.sleep(0.02)
                return b"\0" * min(size, total - offset)
            finally:
                with lock:
                    state["now"] -= 1

        mock_B.dfs_read.side_effect = tracked

        fs._read_pipelined(MagicMock(), 0, total)

        # A literal, not the constant. Comparing against
        # fs._CAT_RANGES_MAX_WORKERS re-read the same value the code uses, so
        # raising the cap to 64 could never fail this.
        assert state["peak"] <= 8, (
            f"peak concurrency {state['peak']} exceeded the executor cap of 8"
        )
        assert fs._CAT_RANGES_MAX_WORKERS == 8, (
            f"the executor cap changed to {fs._CAT_RANGES_MAX_WORKERS}; the "
            "concurrency bound above and the docs both assume 8"
        )
        # And it must actually overlap, or "peak <= 8" is satisfied by 1.
        assert state["peak"] > 1, (
            f"peak concurrency was {state['peak']}: the chunk reads did not overlap"
        )


class TestLockDiscipline:
    def test_cat_file_releases_the_lock_before_the_pipelined_read(self, fs_and_mock):
        """The deadlock shape the rwlock notes warn about.

        _read_pipelined waits on executor workers that each take the shared
        lock. A caller holding the lock across that wait is the one pattern
        that can stall against a waiting writer; find() already drops the
        lock before concurrent.futures.wait() for exactly this reason.

        Measured by sampling the reader count at the instant _read_pipelined is
        entered, which is still single-threaded: 0 means the caller released
        the lock, >=1 means it is holding one across the wait. Sampling from
        inside the workers instead looks tempting but is racy; a worker can
        decrement its own bookkeeping before the enclosing `with` releases its
        read lock, so readers legitimately exceeds the active count.
        """
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        mock_B.dfs_read.side_effect = lambda _d, _o, off, size: b"\0" * min(size, total - off)
        mock_B.dfs_lookup_path.return_value = (
            MagicMock(name="file"),
            SimpleNamespace(size=total, mode=0o100644, type_name="file", is_dir=False),
        )

        readers_on_entry = []
        real = fs._read_pipelined

        def spy(obj, offset, size, *, generation=None):
            readers_on_entry.append(fs._lock.readers)
            return real(obj, offset, size, generation=generation)

        with patch.object(fs, "_read_pipelined", side_effect=spy):
            fs.cat_file("/big.bin")

        assert readers_on_entry, "cat_file never reached the pipelined read"
        assert readers_on_entry == [0], (
            f"cat_file entered the pipelined read holding {readers_on_entry[0]} "
            "read lock(s); it must release before waiting on its own executor"
        )

    def test_fetch_range_releases_the_lock_before_the_pipelined_read(self, fs_and_mock):
        """Same discipline on the fs.open() path."""
        fs, mock_B = fs_and_mock
        total = 6 * CHUNK
        mock_B.dfs_read.side_effect = lambda _d, _o, off, size: b"\0" * min(size, total - off)
        # DFSFile opens via dfs_lookup_rel_path, and AbstractBufferedFile uses
        # the size from it. Leaving the fixture's size=0 here makes read() a
        # no-op and the test passes vacuously on an empty file.
        mock_B.dfs_lookup_rel_path.return_value = (
            MagicMock(name="file"),
            SimpleNamespace(size=total, mode=0o100644, type_name="file", is_dir=False),
        )

        readers_on_entry = []
        real = fs._read_pipelined

        def spy(obj, offset, size, *, generation=None):
            readers_on_entry.append(fs._lock.readers)
            return real(obj, offset, size, generation=generation)

        with patch.object(fs, "_read_pipelined", side_effect=spy), fs.open("/big.bin", "rb") as f:
            f.read(total)

        assert readers_on_entry, "the read never reached the pipelined path"
        assert all(r == 0 for r in readers_on_entry), (
            f"_fetch_range held read lock(s) across the pipelined read: {readers_on_entry}"
        )


@pytest.mark.live
class TestPipelinedLive:
    """Against a real cluster, on both matrix versions."""

    def test_large_read_roundtrips_exactly(self, dfs_fs):
        """64 MB through the pipelined path, byte-for-byte."""
        import os as _os

        path = f"{dfs_fs._test_dir}/pipelined-64mb.bin"
        payload = _os.urandom(64 * 1024 * 1024)
        dfs_fs.pipe_file(path, payload)

        assert dfs_fs.cat_file(path) == payload, "cat_file corrupted a 64 MB read"

        with dfs_fs.open(path, "rb") as f:
            assert f.read() == payload, "open().read() corrupted a 64 MB read"

    def test_ranged_large_read_is_exact(self, dfs_fs):
        """An offset range spanning chunk boundaries."""
        import os as _os

        path = f"{dfs_fs._test_dir}/pipelined-range.bin"
        payload = _os.urandom(48 * 1024 * 1024)
        dfs_fs.pipe_file(path, payload)

        start, end = CHUNK + 1234, 3 * CHUNK + 99
        assert dfs_fs.cat_file(path, start=start, end=end) == payload[start:end]

    def test_read_past_eof_returns_only_real_data(self, dfs_fs):
        """Asking for more than the file holds must not pad or corrupt."""
        path = f"{dfs_fs._test_dir}/pipelined-short.bin"
        payload = b"z" * (2 * CHUNK + 777)
        dfs_fs.pipe_file(path, payload)

        with dfs_fs.open(path, "rb") as f:
            got = f.read(10 * CHUNK)

        assert got == payload, f"expected {len(payload)} bytes, got {len(got)}"


class TestNoNestedSubmission:
    """cat_ranges and _read_pipelined share one executor; that nearly hung."""

    _SCENARIO = """
import sys
sys.path.insert(0, {tests!r})
sys.path.insert(0, {root!r})
from types import SimpleNamespace
from unittest.mock import MagicMock
from conftest import patch_bindings

class E(Exception):
    is_retryable = False

CHUNK = {chunk}
mb = MagicMock()
mb._HAS_DFS_CONNECT = False
mb.DaosError = E
mb.RETRYABLE_ERRORS = ()
mb.CHUNK_SIZE = CHUNK
mb.dfs_lookup_path.return_value = (
    MagicMock(), SimpleNamespace(size=1 << 40, mode=0o100644,
                                 type_name="file", is_dir=False))
mb.dfs_read.side_effect = lambda dfs, obj, off, length: b"\\0" * min(length, CHUNK)

with patch_bindings(mb):
    from daos_fsspec.core import DFSFileSystem
    fs = DFSFileSystem(pool="p", container="c")
    n = {n_paths}
    size = fs._PIPELINE_MIN_BYTES * {size_mult}
    out = fs.cat_ranges([f"/big-{{i}}.bin" for i in range(n)], [0] * n, [size] * n)
    assert len(out) == n, f"expected {{n}} results, got {{len(out)}}"
    assert all(isinstance(r, bytes) for r in out), "non-bytes in result"
    fs.close()
    print("OK")
"""

    def _run_scenario(self, n_paths: int, size_mult: int, deadline: float = 90.0):
        """Run the cat_ranges scenario in a child process; fail if it stalls."""
        root = str(pathlib.Path(__file__).resolve().parent.parent)
        program = self._SCENARIO.format(
            tests=str(pathlib.Path(__file__).resolve().parent),
            root=root,
            chunk=CHUNK,
            n_paths=n_paths,
            size_mult=size_mult,
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                timeout=deadline,
                cwd=root,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"cat_ranges({n_paths} paths) made no progress in "
                f"{deadline:.0f}s; a pipelined read submitted into the "
                "executor it was already running on. The child had to be "
                "killed; its worker threads are non-daemon and unjoinable."
            )
        assert result.returncode == 0, (
            f"scenario failed (exit {result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        assert "OK" in result.stdout, f"scenario did not complete: {result.stdout!r}"

    def test_cat_ranges_over_large_ranges_does_not_deadlock(self):
        """Exactly saturate the pool with reads that all want to pipeline."""
        self._run_scenario(n_paths=8, size_mult=2)

    def test_more_paths_than_workers_does_not_deadlock(self):
        """Oversubscribed: tasks queue behind tasks that must not block."""
        self._run_scenario(n_paths=24, size_mult=1)

    def test_a_direct_large_read_still_pipelines(self, fs_and_mock):
        """The guard must only fire on the pool, not disable pipelining.

        Without this, "fix the deadlock by never pipelining" would pass the
        tests above while silently deleting the feature.
        """
        fs, mock_B = fs_and_mock
        size = fs._PIPELINE_MIN_BYTES * 2
        mock_B.dfs_read.side_effect = lambda dfs, obj, off, length: b"\0" * min(length, CHUNK)

        fs.cat_file("/big.bin", 0, size)

        assert mock_B.dfs_read.call_count >= size // CHUNK, (
            f"expected at least {size // CHUNK} chunk reads, got "
            f"{mock_B.dfs_read.call_count}; the read was not pipelined"
        )

    def test_the_pool_marker_is_cleared_after_each_task(self, fs_and_mock):
        """A leaked marker would silently disable pipelining for that thread."""
        fs, mock_B = fs_and_mock
        mock_B.dfs_read.side_effect = lambda dfs, obj, off, length: b"\0" * min(length, CHUNK)
        n = fs._CAT_RANGES_MAX_WORKERS

        fs.cat_ranges([f"/f{i}.bin" for i in range(n)], [0] * n, [1024] * n)

        assert getattr(fs._on_read_pool, "active", False) is False, (
            "the calling thread was left marked as a pool worker"
        )


def test_every_executor_submission_site_consults_the_pool_marker():
    """Structural guard against the nested-submission deadlock returning."""
    import ast
    import inspect

    from daos_fsspec import core

    tree = ast.parse(inspect.getsource(core))
    offenders = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.unparse(node)
        submits = "self._executor.submit" in body or "self._executor.map" in body
        if submits and "_on_read_pool" not in body:
            offenders.append(node.name)

    assert not offenders, (
        "these functions submit to self._executor without consulting "
        f"_on_read_pool: {sorted(offenders)}. A read running on that executor "
        "must not submit into it and block; that is the cat_ranges deadlock. "
        "Either set the marker (as cat_ranges does) or check it (as "
        "_read_pipelined does)."
    )


def test_the_marker_guard_would_catch_a_new_submission_site():
    """The guard above must not be vacuous.

     Parses a synthetic function that submits without the marker and checks the
     same rule rejects it. Without this, a guard that silently matched nothing
    ; a renamed executor attribute, say; would read as "no offenders".
    """
    import ast

    synthetic = ast.parse(
        "def _new_parallel_read(self):\n"
        "    futures = [self._executor.submit(f) for f in work]\n"
        "    return [f.result() for f in futures]\n"
    )
    node = synthetic.body[0]
    body = ast.unparse(node)

    submits = "self._executor.submit" in body or "self._executor.map" in body
    assert submits, "the rule no longer recognises a submission site"
    assert "_on_read_pool" not in body
