# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for benchmarks."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"


_MB = 1_000_000


_BANDWIDTH_FLOOR = 8 * _MB


def _daos_version() -> str:
    """DAOS version under test, for keying results. 'unknown' if unset."""
    return os.environ.get("DAOS_VERSION", "unknown")


def _baselines_file() -> Path:
    """Per-version baselines path, so matrix runs do not clobber each other."""
    return BENCHMARKS_DIR / f"baselines-{_daos_version()}.json"


def _load_baselines() -> dict:
    """Load existing baselines for this DAOS version, or a fresh skeleton."""
    path = _baselines_file()
    if path.exists():
        return json.loads(path.read_text())
    return {
        "context": {
            "daos_version": _daos_version(),
            "scope": "per-process, single rank",
            "bandwidth_unit": "decimal MB (10**6 bytes)",
            "samples": "single measurement per figure; not a median",
        },
        "results": {},
    }


def _save_baselines(baselines: dict) -> None:
    """Save baselines for this DAOS version."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    _baselines_file().write_text(json.dumps(baselines, indent=2) + "\n")


def _record(
    name: str,
    elapsed: float,
    *,
    ops: int | None = None,
    nbytes: int | None = None,
    op_unit: str = "ops_per_sec",
) -> dict:
    """Record one benchmark result and return the recorded dict."""
    entry: dict = {"elapsed_s": round(elapsed, 3)}
    if ops is not None:
        entry[op_unit] = round(ops / elapsed, 1)
    if nbytes is not None:
        entry["bytes"] = nbytes
        if nbytes >= _BANDWIDTH_FLOOR:
            entry["throughput_mb_s"] = round((nbytes / elapsed) / _MB, 1)
        else:
            entry["throughput_mb_s"] = None
            entry["throughput_omitted"] = (
                f"latency-bound: {nbytes} B is below the {_BANDWIDTH_FLOOR} B "
                "floor where MB/s describes bandwidth rather than round-trip latency"
            )

    baselines = _load_baselines()
    baselines["results"][name] = entry
    _save_baselines(baselines)
    return entry


@pytest.mark.live
class BenchmarkSmallFiles:
    """Benchmark small file operations."""

    def test_bench_pipe_100_files(self, dfs_fs):
        """Benchmark: pipe 100 small files (1KB each)."""
        d = dfs_fs._test_dir
        n = 100
        size = 1024
        data = os.urandom(size)

        dfs_fs.mkdir(f"{d}/bench-pipe-100")
        # Distinct content per file so the read-back catches a path-collision bug
        # (e.g. all writes landing on one path) that identical data would hide.
        payloads = [f"{i:04d}:".encode() + data[: size - 5] for i in range(n)]

        start = time.monotonic()
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/bench-pipe-100/f-{i:04d}.bin", payloads[i])
        elapsed = time.monotonic() - start

        # Verify every file was written with its own correct content.
        for i in range(n):
            assert dfs_fs.cat(f"{d}/bench-pipe-100/f-{i:04d}.bin") == payloads[i]

        # Latency-bound: 100 kB total. op/s only; see methodology rule 1.
        rec = _record("pipe_100x1KB", elapsed, ops=n, nbytes=n * size)
        print(f"\n  pipe 100x1KB: {elapsed:.3f}s ({rec['ops_per_sec']:.0f} ops/s, per process)")

    def test_bench_cat_100_files(self, dfs_fs):
        """Benchmark: cat 100 small files (1KB each)."""
        d = dfs_fs._test_dir
        n = 100
        size = 1024
        data = os.urandom(size)

        dfs_fs.mkdir(f"{d}/bench-cat-100")
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/bench-cat-100/f-{i:04d}.bin", data)

        start = time.monotonic()
        for i in range(n):
            result = dfs_fs.cat(f"{d}/bench-cat-100/f-{i:04d}.bin")
            assert result == data  # full content, not just length
        elapsed = time.monotonic() - start

        rec = _record("cat_100x1KB", elapsed, ops=n, nbytes=n * size)
        print(f"\n  cat 100x1KB: {elapsed:.3f}s ({rec['ops_per_sec']:.0f} ops/s, per process)")

    def test_bench_ls_500_entries(self, dfs_fs):
        """Benchmark: ls on directory with 500 entries."""
        d = dfs_fs._test_dir
        n = 500

        dfs_fs.mkdir(f"{d}/bench-ls-500")
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/bench-ls-500/f-{i:04d}.txt", b"x")

        start = time.monotonic()
        entries = dfs_fs.ls(f"{d}/bench-ls-500", detail=True)
        elapsed = time.monotonic() - start

        assert len(entries) == n

        rec = _record("ls_500_entries", elapsed, ops=n, op_unit="entries_per_sec")
        print(
            f"\n  ls(detail=True) 500 entries: {elapsed:.3f}s "
            f"({rec['entries_per_sec']:.1f} ent/s, per process)"
        )


@pytest.mark.live
class BenchmarkLargeFiles:
    """Benchmark large file I/O."""

    def test_bench_write_16mb(self, dfs_fs):
        """Benchmark: write 16MB file."""
        d = dfs_fs._test_dir
        path = f"{d}/bench-write-16mb.bin"
        size = 16 * 1024 * 1024
        data = os.urandom(size)

        start = time.monotonic()
        with dfs_fs.open(path, "wb") as f:
            f.write(data)
        elapsed = time.monotonic() - start

        # Verify the write actually landed correctly (outside the timed region) -
        # a benchmark that only times a write proves nothing if bytes are dropped.
        assert dfs_fs.cat(path) == data

        rec = _record("write_16MB", elapsed, ops=1, nbytes=size)
        print(f"\n  write 16MB: {elapsed:.3f}s ({rec['throughput_mb_s']:.0f} MB/s, per process)")

    def test_bench_read_16mb(self, dfs_fs):
        """Benchmark: read 16MB file."""
        d = dfs_fs._test_dir
        path = f"{d}/bench-read-16mb.bin"
        size = 16 * 1024 * 1024
        data = os.urandom(size)

        with dfs_fs.open(path, "wb") as f:
            f.write(data)

        start = time.monotonic()
        with dfs_fs.open(path, "rb") as f:
            result = f.read()
        elapsed = time.monotonic() - start

        # Full content compare, not just length: a garbled/stale read that
        # returns the right number of bytes must still fail.
        assert result == data
        rec = _record("read_16MB", elapsed, ops=1, nbytes=size)
        print(f"\n  read 16MB: {elapsed:.3f}s ({rec['throughput_mb_s']:.0f} MB/s, per process)")

    def test_bench_write_64mb(self, dfs_fs):
        """Benchmark: write 64MB file (crosses 16MB chunk boundary)."""
        d = dfs_fs._test_dir
        path = f"{d}/bench-write-64mb.bin"
        size = 64 * 1024 * 1024
        data = os.urandom(size)

        start = time.monotonic()
        with dfs_fs.open(path, "wb") as f:
            f.write(data)
        elapsed = time.monotonic() - start

        # This file crosses the 16MB chunk boundary: verify the WHOLE thing
        # round-trips, so a chunk-boundary truncation/corruption can't pass.
        assert dfs_fs.cat(path) == data

        rec = _record("write_64MB", elapsed, ops=1, nbytes=size)
        print(f"\n  write 64MB: {elapsed:.3f}s ({rec['throughput_mb_s']:.0f} MB/s, per process)")


@pytest.mark.live
class BenchmarkMetadata:
    """Benchmark metadata operations."""

    def test_bench_stat_100_files(self, dfs_fs):
        """Benchmark: stat 100 files."""
        d = dfs_fs._test_dir
        n = 100

        dfs_fs.mkdir(f"{d}/bench-stat-100")
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/bench-stat-100/f-{i:04d}.txt", b"x")

        start = time.monotonic()
        for i in range(n):
            dfs_fs.info(f"{d}/bench-stat-100/f-{i:04d}.txt")
        elapsed = time.monotonic() - start

        rec = _record("stat_100_files", elapsed, ops=n)
        print(f"\n  stat 100 files: {elapsed:.3f}s ({rec['ops_per_sec']:.0f} ops/s, per process)")

    def test_bench_exists_100_files(self, dfs_fs):
        """Benchmark: exists check on 100 files."""
        d = dfs_fs._test_dir
        n = 100

        dfs_fs.mkdir(f"{d}/bench-exists-100")
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/bench-exists-100/f-{i:04d}.txt", b"x")

        start = time.monotonic()
        for i in range(n):
            assert dfs_fs.exists(f"{d}/bench-exists-100/f-{i:04d}.txt")
        elapsed = time.monotonic() - start

        rec = _record("exists_100_files", elapsed, ops=n)
        print(f"\n  exists 100 files: {elapsed:.3f}s ({rec['ops_per_sec']:.0f} ops/s, per process)")

    def test_bench_mkdir_50_dirs(self, dfs_fs):
        """Benchmark: mkdir 50 directories."""
        d = dfs_fs._test_dir
        n = 50

        dfs_fs.mkdir(f"{d}/bench-mkdir-50")

        start = time.monotonic()
        for i in range(n):
            dfs_fs.mkdir(f"{d}/bench-mkdir-50/dir-{i:03d}")
        elapsed = time.monotonic() - start

        rec = _record("mkdir_50_dirs", elapsed, ops=n)
        print(f"\n  mkdir 50 dirs: {elapsed:.3f}s ({rec['ops_per_sec']:.0f} ops/s, per process)")


@pytest.mark.live
class BenchmarkConcurrency:
    """Benchmark concurrent operations."""

    def test_bench_concurrent_reads_8_threads(self, dfs_fs):
        """Benchmark: 8 threads reading simultaneously."""
        import concurrent.futures

        d = dfs_fs._test_dir
        n = 8
        size = 64 * 1024
        # Distinct content per file (unique 16-byte prefix) so a thread reading
        # another thread's file under concurrency is detectable: identical data
        # plus a length-only check would hide exactly that bug.
        payloads = [f"file-{i:02d}-".encode() + os.urandom(size - 8) for i in range(n)]

        dfs_fs.mkdir(f"{d}/bench-concurrent-8")
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/bench-concurrent-8/f-{i}.bin", payloads[i])

        def read_file(idx):
            return idx, dfs_fs.cat(f"{d}/bench-concurrent-8/f-{idx}.bin")

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(read_file, i) for i in range(n)]
            results = [f.result() for f in futures]
        elapsed = time.monotonic() - start

        for idx, result in results:
            assert result == payloads[idx], f"file {idx} returned wrong content"

        # 8 x 64 kB is 512 kB; below the bandwidth floor, so this is an
        # 8-way concurrent round-trip measurement, not a throughput one. What
        # it is actually testing is whether the shared lock lets the 8 reads
        # overlap; op/s is the figure that shows that.
        rec = _record("concurrent_reads_8x64KB", elapsed, ops=n, nbytes=n * size)
        print(
            f"\n  concurrent reads 8x64KB: {elapsed:.3f}s "
            f"({rec['ops_per_sec']:.0f} reads/s, 8 threads, per process)"
        )


class TestBenchmarkRecording:
    """Offline tests for the recording helpers.

    Every Benchmark* class above is @pytest.mark.live, so without these the
    _record logic; which decides whether a MB/s figure is meaningful at all --
    would ship with no coverage and no way to fail.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        """Redirect writes to tmp_path so tests never touch benchmarks/.

        Patches the module object directly rather than by dotted string:
        there is no tests/__init__.py, so a "tests.test_benchmarks..." target
        resolves a *second* copy of this module and the real module global is
        left alone; which silently wrote into the repo's benchmarks/ dir.
        """
        monkeypatch.setattr(sys.modules[__name__], "BENCHMARKS_DIR", tmp_path)
        monkeypatch.delenv("DAOS_VERSION", raising=False)

    def test_latency_bound_transfer_suppresses_throughput(self, tmp_path):
        """A sub-floor transfer must NOT get a MB/s figure."""
        rec = _record("tiny", elapsed=1.6, ops=500, nbytes=500 * 128)

        assert rec["throughput_mb_s"] is None
        assert "throughput_omitted" in rec
        assert rec["ops_per_sec"] == 312.5

    def test_bandwidth_bound_transfer_reports_throughput(self, tmp_path):
        """At or above the floor, MB/s is derived; in DECIMAL MB.

        The byte count is a literal, deliberately not ``64 * _MB``. Expressed
        in terms of _MB the assertion holds under either divisor and the test
        cannot detect a unit regression at all; it was written that way first
        and passed against the MiB code it was supposed to catch.
        """
        rec = _record("big", elapsed=2.0, ops=1, nbytes=64_000_000)

        assert rec["throughput_mb_s"] == 32.0
        assert "throughput_omitted" not in rec

    def test_floor_boundary_is_inclusive(self, tmp_path):
        """Exactly at the floor counts as bandwidth-bound.

        Literal byte counts again, so this also pins the floor at 8e6 B rather
        than 8 MiB; ``_BANDWIDTH_FLOOR`` is derived from ``_MB``, so writing
        it that way would track a unit change instead of catching it.
        """
        assert _BANDWIDTH_FLOOR == 8_000_000
        at_floor = _record("at", elapsed=1.0, nbytes=8_000_000)
        below = _record("below", elapsed=1.0, nbytes=7_999_999)

        assert at_floor["throughput_mb_s"] == 8.0
        assert below["throughput_mb_s"] is None

    def test_op_unit_is_overridable_for_listings(self, tmp_path):
        """op unit is overridable for listings."""
        rec = _record("ls", elapsed=5.525, ops=500, op_unit="entries_per_sec")

        assert "ops_per_sec" not in rec

        assert rec["entries_per_sec"] == 90.5

    def test_metadata_only_op_has_no_byte_or_throughput_key(self, tmp_path):
        """stat/exists/mkdir move no payload, so neither key should appear."""
        rec = _record("stat", elapsed=1.343, ops=100)

        assert "bytes" not in rec
        assert "throughput_mb_s" not in rec

        assert rec["ops_per_sec"] == 74.5

    def test_results_are_keyed_by_daos_version(self, tmp_path, monkeypatch):
        """Two matrix versions must not overwrite each other."""
        monkeypatch.setenv("DAOS_VERSION", "2.6.5")
        _record("write_64MB", elapsed=2.0, nbytes=64 * _MB)
        monkeypatch.setenv("DAOS_VERSION", "2.8.0")
        _record("write_64MB", elapsed=1.0, nbytes=64 * _MB)

        v265 = json.loads((tmp_path / "baselines-2.6.5.json").read_text())
        v280 = json.loads((tmp_path / "baselines-2.8.0.json").read_text())

        assert v265["results"]["write_64MB"]["throughput_mb_s"] == 32.0
        assert v280["results"]["write_64MB"]["throughput_mb_s"] == 64.0
        assert v265["context"]["daos_version"] == "2.6.5"
        assert v280["context"]["daos_version"] == "2.8.0"

    def test_context_block_records_scope_and_unit(self, tmp_path):
        """Scope and unit must be in the file, not just in a docstring."""
        _record("whatever", elapsed=1.0, ops=1)
        written = json.loads((tmp_path / "baselines-unknown.json").read_text())

        assert written["context"]["scope"] == "per-process, single rank"
        assert "10**6" in written["context"]["bandwidth_unit"]
        assert "single measurement" in written["context"]["samples"]

    def test_repeated_records_accumulate(self, tmp_path):
        """Recording a second benchmark must not drop the first."""
        _record("first", elapsed=1.0, ops=10)
        _record("second", elapsed=2.0, ops=20)

        written = json.loads((tmp_path / "baselines-unknown.json").read_text())
        assert set(written["results"]) == {"first", "second"}
