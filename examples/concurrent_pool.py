#!/usr/bin/env python3
"""
Example 3: Concurrent access from many threads via DFSFileSystemPool.

Each pooled connection has its own DFS mount and lock. The pool limits how
many workers can borrow a connection at once. Individual file objects must
not be shared between threads without application synchronization.

This writes and reads data from 16 threads and verifies every byte.

Run:
    export DAOS_POOL=mypool DAOS_CONTAINER=mycont
    python examples/concurrent_pool.py
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from daos_fsspec.pool import DFSFileSystemPool

POOL = os.environ.get("DAOS_POOL")
CONT = os.environ.get("DAOS_CONTAINER")
if not (POOL and CONT):
    sys.exit("set DAOS_POOL and DAOS_CONTAINER first")

BASE = "/concurrent-example"
N_THREADS = 16
FILES_PER_THREAD = 20
FILE_SIZE = 256 * 1024  # 256 KiB each


def payload(tid: int, i: int) -> bytes:
    return (f"t{tid}-f{i}-".encode() * (FILE_SIZE // 8))[:FILE_SIZE]


def writer(pool, tid: int) -> int:
    with pool.connection() as fs:
        for i in range(FILES_PER_THREAD):
            fs.pipe_file(f"{BASE}/t{tid:02d}/f{i:03d}.bin", payload(tid, i))
    return FILES_PER_THREAD


def reader(pool, tid: int) -> int:
    with pool.connection() as fs:
        for i in range(FILES_PER_THREAD):
            data = fs.cat_file(f"{BASE}/t{tid:02d}/f{i:03d}.bin")
            if data != payload(tid, i):
                raise AssertionError(f"corruption in t{tid}/f{i}")
    return FILES_PER_THREAD


def main() -> int:
    # Pool smaller than the thread count on purpose: exercises acquire/release
    # contention, which is the whole point of a pool.
    pool = DFSFileSystemPool(pool=POOL, container=CONT, size=6)
    print(
        f"[pool] size={pool.status()['size']}, {N_THREADS} threads x "
        f"{FILES_PER_THREAD} files x {FILE_SIZE // 1024} KiB"
    )

    with pool.connection() as fs:
        if fs.exists(BASE):
            fs.rm(BASE, recursive=True)
        for tid in range(N_THREADS):
            fs.mkdir(f"{BASE}/t{tid:02d}", create_parents=True)

    total_bytes = N_THREADS * FILES_PER_THREAD * FILE_SIZE

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        written = sum(
            f.result() for f in as_completed([ex.submit(writer, pool, t) for t in range(N_THREADS)])
        )
    dt = time.time() - t0
    print(
        f"[pool] wrote {written} files ({total_bytes / 1e6:.0f} MB) "
        f"in {dt:.2f}s = {total_bytes / 1e6 / dt:.0f} MB/s"
    )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        read_ok = sum(
            f.result() for f in as_completed([ex.submit(reader, pool, t) for t in range(N_THREADS)])
        )
    dt = time.time() - t0
    print(
        f"[pool] verified {read_ok} files in {dt:.2f}s "
        f"= {total_bytes / 1e6 / dt:.0f} MB/s, zero corruption"
    )

    st = pool.status()
    assert st["in_use"] == 0, f"leaked connections: {st}"
    assert pool.health_check(), "pool health_check failed"
    print(f"[pool] status after run: {st}")

    pool.close()
    print("[pool] OK: pool handled 16-way concurrency, no corruption or leaks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
