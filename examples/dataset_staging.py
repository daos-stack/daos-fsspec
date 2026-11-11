#!/usr/bin/env python3
"""Example 1: Stage a sharded parquet dataset into DAOS and read it back."""

import os
import sys
import time

import pandas as pd

POOL = os.environ.get("DAOS_POOL")
CONT = os.environ.get("DAOS_CONTAINER")
if not (POOL and CONT):
    sys.exit("set DAOS_POOL and DAOS_CONTAINER first")

BASE = f"dfs://{POOL}/{CONT}/datasets/sensor-v1/train"
N_SHARDS = 8
ROWS_PER_SHARD = 50_000


def main() -> int:
    print(f"[dataset] staging {N_SHARDS} parquet shards -> {BASE}")

    import fsspec

    fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)

    # NOTE: unlike object stores (s3://), DAOS DFS has *real* directories, and
    # writing a file to a nested path does not auto-create them: pandas/pyarrow
    # call fs.open(path, "wb"), which raises ENOENT if the parent is missing.
    # Create the tree first.
    fs.makedirs("/datasets/sensor-v1/train", exist_ok=True)

    # 1. Write shards straight to DAOS via the dfs:// URL (no local staging).
    t0 = time.time()
    total_written = 0
    for i in range(N_SHARDS):
        df = pd.DataFrame(
            {
                "id": range(i * ROWS_PER_SHARD, (i + 1) * ROWS_PER_SHARD),
                "temp": [20.0 + (n % 100) * 0.1 for n in range(ROWS_PER_SHARD)],
                "label": [n % 2 for n in range(ROWS_PER_SHARD)],
            }
        )
        df.to_parquet(f"{BASE}/shard-{i:05d}.parquet", index=False)
        total_written += len(df)
    print(f"[dataset] wrote {total_written:,} rows in {time.time() - t0:.2f}s")

    # 2. Discover the shards with a glob, the way a data loader would.
    shards = fs.glob("/datasets/sensor-v1/train/*.parquet")
    print(f"[dataset] glob found {len(shards)} shards")
    assert len(shards) == N_SHARDS, f"expected {N_SHARDS}, glob saw {len(shards)}"

    # 3. Read every shard back with pandas and check the data survived.
    t0 = time.time()
    full = pd.concat(
        [pd.read_parquet(f"dfs://{POOL}/{CONT}{s}") for s in shards],
        ignore_index=True,
    )
    print(f"[dataset] read back {len(full):,} rows in {time.time() - t0:.2f}s")

    assert len(full) == total_written, "row count mismatch after round-trip"
    assert full["id"].nunique() == total_written, "duplicate/lost ids"
    assert full["label"].isin([0, 1]).all(), "label column corrupted"


    detail = fs.ls("/datasets/sensor-v1/train/", detail=True)
    assert all(d["size"] > 0 for d in detail), "zero-byte shards"

    print("[dataset] OK: parquet dataset round-tripped through DAOS cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
