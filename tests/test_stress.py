# Copyright 2026 Enakta Labs Ltd
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for stress."""

import concurrent.futures
import os
import time

import pytest


@pytest.mark.live
class TestStressLarge:
    """Large-scale stress tests."""

    def test_1000_files(self, dfs_fs):
        """Create and read 1000 small files."""
        d = dfs_fs._test_dir
        n = 1000
        dfs_fs.mkdir(f"{d}/stress-1000")

        # Create
        start = time.monotonic()
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/stress-1000/f-{i:04d}.txt", f"data-{i}".encode())
        create_time = time.monotonic() - start

        # List
        start = time.monotonic()
        entries = dfs_fs.ls(f"{d}/stress-1000")
        list_time = time.monotonic() - start

        assert len(entries) == n

        # Read all
        start = time.monotonic()
        for i in range(n):
            result = dfs_fs.cat(f"{d}/stress-1000/f-{i:04d}.txt")
            assert result == f"data-{i}".encode()
        read_time = time.monotonic() - start

        # Log performance
        print(
            f"\n1000 files: create={create_time:.2f}s, list={list_time:.2f}s, read={read_time:.2f}s"
        )

    def test_concurrent_reads_16_threads(self, dfs_fs):
        """16 threads reading different files simultaneously."""
        d = dfs_fs._test_dir
        n = 16
        dfs_fs.mkdir(f"{d}/concurrent-16")

        # Create test files
        for i in range(n):
            dfs_fs.pipe_file(f"{d}/concurrent-16/file-{i}.txt", f"thread-data-{i}".encode())

        def read_file(idx):
            data = dfs_fs.cat(f"{d}/concurrent-16/file-{idx}.txt")
            return data

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(read_file, i) for i in range(n)]
            results = [f.result() for f in futures]
        elapsed = time.monotonic() - start

        for i, result in enumerate(results):
            assert result == f"thread-data-{i}".encode()

        print(f"\n16 concurrent reads: {elapsed:.2f}s")

    def test_concurrent_writes_16_threads(self, dfs_fs):
        """16 threads writing different files simultaneously."""
        d = dfs_fs._test_dir
        n = 16
        dfs_fs.mkdir(f"{d}/concurrent-write-16")

        def write_file(idx):
            data = os.urandom(64 * 1024)  # 64KB each
            path = f"{d}/concurrent-write-16/cw-{idx}.bin"
            dfs_fs.pipe_file(path, data)
            return path, data

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(write_file, i) for i in range(n)]
            results = [f.result() for f in futures]
        elapsed = time.monotonic() - start

        # Verify all files
        for path, expected in results:
            actual = dfs_fs.cat(path)
            assert actual == expected

        print(f"\n16 concurrent writes (64KB each): {elapsed:.2f}s")

    def test_large_file_64mb(self, dfs_fs):
        """Write and read 64MB file."""
        d = dfs_fs._test_dir
        path = f"{d}/large-64mb.bin"
        data = os.urandom(64 * 1024 * 1024)

        start = time.monotonic()
        with dfs_fs.open(path, "wb") as f:
            f.write(data)
        write_time = time.monotonic() - start

        start = time.monotonic()
        with dfs_fs.open(path, "rb") as f:
            result = f.read()
        read_time = time.monotonic() - start

        assert len(result) == len(data)
        assert result == data

        print(f"\n64MB file: write={write_time:.2f}s, read={read_time:.2f}s")

    def test_deep_directory_50_levels(self, dfs_fs):
        """Create a 50-level deep directory tree."""
        d = dfs_fs._test_dir
        n = 50
        deep = "/".join([f"level-{i}" for i in range(n)])
        path = f"{d}/deep-50/{deep}"

        start = time.monotonic()
        dfs_fs.mkdir(path, create_parents=True)
        create_time = time.monotonic() - start

        assert dfs_fs.exists(path)

        # Write and read at the bottom
        dfs_fs.pipe_file(f"{path}/bottom.txt", b"deep-data")
        result = dfs_fs.cat(f"{path}/bottom.txt")
        assert result == b"deep-data"

        print(f"\n50-level deep mkdir: {create_time:.2f}s")

    def test_wide_directory_500_entries(self, dfs_fs):
        """ls on directory with 500 entries."""
        d = dfs_fs._test_dir
        n = 500
        dfs_fs.mkdir(f"{d}/wide-500")

        for i in range(n):
            dfs_fs.pipe_file(f"{d}/wide-500/f-{i:04d}.txt", f"data-{i}".encode())

        start = time.monotonic()
        entries = dfs_fs.ls(f"{d}/wide-500", detail=True)
        elapsed = time.monotonic() - start

        assert len(entries) == n
        print(f"\n500-entry ls(detail=True): {elapsed:.2f}s")

    def test_rapid_create_delete_100_cycles(self, dfs_fs):
        """Rapidly create and delete the same file 100 times."""
        d = dfs_fs._test_dir
        path = f"{d}/rapid-100.txt"
        n = 100

        start = time.monotonic()
        for i in range(n):
            dfs_fs.pipe_file(path, f"iteration-{i}".encode())
            assert dfs_fs.exists(path)
            dfs_fs.rm(path)
            assert not dfs_fs.exists(path)
        elapsed = time.monotonic() - start

        print(f"\n100 create/delete cycles: {elapsed:.2f}s")


@pytest.mark.live
class TestStressPool:
    """Stress tests using connection pool."""

    def test_pool_concurrent_access(self, daos_pool, daos_container):
        """Multiple threads using connection pool simultaneously."""
        from daos_fsspec.pool import DFSFileSystemPool

        pool_size = 4
        n_threads = 16
        n_files = 100

        with DFSFileSystemPool(
            pool=daos_pool, container=daos_container, size=pool_size
        ) as conn_pool:
            # Create test directory
            fs = conn_pool.acquire()
            test_dir = f"/pool-stress-{os.urandom(4).hex()}"
            fs.mkdir(test_dir)
            conn_pool.release(fs)

            def worker(idx):
                with conn_pool.connection() as fs:
                    fs.mkdir(f"{test_dir}/worker-{idx}")
                    for i in range(n_files):
                        path = f"{test_dir}/worker-{idx}/file-{i}.txt"
                        fs.pipe_file(path, f"data-{idx}-{i}".encode())
                    return n_files

            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(n_threads)]
                results = [f.result() for f in futures]
            elapsed = time.monotonic() - start

            total_files = sum(results)
            print(
                f"\nPool stress: {n_threads} threads × {n_files} files = {total_files} in {elapsed:.2f}s"
            )

            # Cleanup
            fs = conn_pool.acquire()
            try:
                fs.rm(test_dir, recursive=True)
            finally:
                conn_pool.release(fs)
