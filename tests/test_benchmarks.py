# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for daos-fsspec.

These tests measure and record performance baselines.
Requires a running DAOS cluster.

Run with: pytest tests/test_benchmarks.py -v -s
"""

import json
import os
import time
from pathlib import Path

import pytest

BENCHMARKS_FILE = Path(__file__).parent.parent / "benchmarks" / "baselines.json"


def _load_baselines() -> dict:
    """Load existing baselines or return empty dict."""
    if BENCHMARKS_FILE.exists():
        return json.loads(BENCHMARKS_FILE.read_text())
    return {}


def _save_baselines(baselines: dict):
    """Save baselines to file."""
    BENCHMARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_FILE.write_text(json.dumps(baselines, indent=2) + "\n")


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

        ops_per_sec = n / elapsed
        print(f"\n  pipe 100x1KB: {elapsed:.3f}s ({ops_per_sec:.0f} ops/s)")

        baselines = _load_baselines()
        baselines["pipe_100x1KB"] = {
            "ops_per_sec": round(ops_per_sec, 1),
            "elapsed_s": round(elapsed, 3),
        }
        _save_baselines(baselines)

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

        ops_per_sec = n / elapsed
        throughput_mb = (n * size / elapsed) / (1024 * 1024)
        print(
            f"\n  cat 100x1KB: {elapsed:.3f}s ({ops_per_sec:.0f} ops/s, {throughput_mb:.1f} MB/s)"
        )

        baselines = _load_baselines()
        baselines["cat_100x1KB"] = {
            "ops_per_sec": round(ops_per_sec, 1),
            "throughput_mb_s": round(throughput_mb, 1),
        }
        _save_baselines(baselines)

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
        print(f"\n  ls(detail=True) 500 entries: {elapsed:.3f}s")

        baselines = _load_baselines()
        baselines["ls_500_entries"] = {"elapsed_s": round(elapsed, 3)}
        _save_baselines(baselines)


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

        # Verify the write actually landed correctly (outside the timed region) —
        # a benchmark that only times a write proves nothing if bytes are dropped.
        assert dfs_fs.cat(path) == data

        throughput_mb = (size / elapsed) / (1024 * 1024)
        print(f"\n  write 16MB: {elapsed:.3f}s ({throughput_mb:.0f} MB/s)")

        baselines = _load_baselines()
        baselines["write_16MB"] = {
            "elapsed_s": round(elapsed, 3),
            "throughput_mb_s": round(throughput_mb, 1),
        }
        _save_baselines(baselines)

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

        # Full content compare, not just length — a garbled/stale read that
        # returns the right number of bytes must still fail.
        assert result == data
        throughput_mb = (size / elapsed) / (1024 * 1024)
        print(f"\n  read 16MB: {elapsed:.3f}s ({throughput_mb:.0f} MB/s)")

        baselines = _load_baselines()
        baselines["read_16MB"] = {
            "elapsed_s": round(elapsed, 3),
            "throughput_mb_s": round(throughput_mb, 1),
        }
        _save_baselines(baselines)

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

        # This file crosses the 16MB chunk boundary — verify the WHOLE thing
        # round-trips, so a chunk-boundary truncation/corruption can't pass.
        assert dfs_fs.cat(path) == data

        throughput_mb = (size / elapsed) / (1024 * 1024)
        print(f"\n  write 64MB: {elapsed:.3f}s ({throughput_mb:.0f} MB/s)")

        baselines = _load_baselines()
        baselines["write_64MB"] = {
            "elapsed_s": round(elapsed, 3),
            "throughput_mb_s": round(throughput_mb, 1),
        }
        _save_baselines(baselines)


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

        ops_per_sec = n / elapsed
        print(f"\n  stat 100 files: {elapsed:.3f}s ({ops_per_sec:.0f} ops/s)")

        baselines = _load_baselines()
        baselines["stat_100_files"] = {"ops_per_sec": round(ops_per_sec, 1)}
        _save_baselines(baselines)

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

        ops_per_sec = n / elapsed
        print(f"\n  exists 100 files: {elapsed:.3f}s ({ops_per_sec:.0f} ops/s)")

        baselines = _load_baselines()
        baselines["exists_100_files"] = {"ops_per_sec": round(ops_per_sec, 1)}
        _save_baselines(baselines)

    def test_bench_mkdir_50_dirs(self, dfs_fs):
        """Benchmark: mkdir 50 directories."""
        d = dfs_fs._test_dir
        n = 50

        dfs_fs.mkdir(f"{d}/bench-mkdir-50")

        start = time.monotonic()
        for i in range(n):
            dfs_fs.mkdir(f"{d}/bench-mkdir-50/dir-{i:03d}")
        elapsed = time.monotonic() - start

        ops_per_sec = n / elapsed
        print(f"\n  mkdir 50 dirs: {elapsed:.3f}s ({ops_per_sec:.0f} ops/s)")

        baselines = _load_baselines()
        baselines["mkdir_50_dirs"] = {"ops_per_sec": round(ops_per_sec, 1)}
        _save_baselines(baselines)


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
        # another thread's file under concurrency is detectable — identical data
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

        throughput_mb = (n * size / elapsed) / (1024 * 1024)
        print(f"\n  concurrent reads 8x64KB: {elapsed:.3f}s ({throughput_mb:.0f} MB/s)")

        baselines = _load_baselines()
        baselines["concurrent_reads_8x64KB"] = {
            "elapsed_s": round(elapsed, 3),
            "throughput_mb_s": round(throughput_mb, 1),
        }
        _save_baselines(baselines)
