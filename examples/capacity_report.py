#!/usr/bin/env python3
"""Example 4: A capacity + usage report for a DAOS container."""

import os
import sys

from daos_fsspec.metrics import DFSFileSystemWithMetrics

POOL = os.environ.get("DAOS_POOL")
CONT = os.environ.get("DAOS_CONTAINER")
if not (POOL and CONT):
    sys.exit("set DAOS_POOL and DAOS_CONTAINER first")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> int:
    fs = DFSFileSystemWithMetrics(pool=POOL, container=CONT)

    # 1. is the connection alive?
    assert fs.health_check(), "DAOS connection is dead"
    print("[report] health_check: OK")

    # 2. pool-level capacity (SCM + NVMe tiers)
    space = fs.pool_space()
    fields = {
        k: getattr(space, k)
        for k in dir(space)
        if not k.startswith("_") and isinstance(getattr(space, k, None), (int, float))
    }
    print("[report] pool space:")
    for k, v in sorted(fields.items()):
        print(f"         {k:>16} = {human(v) if v > 1024 else v}")
    # free must never exceed total, nothing negative
    totals = [v for k, v in fields.items() if "total" in k.lower()]
    frees = [v for k, v in fields.items() if "free" in k.lower()]
    assert all(v >= 0 for v in fields.values()), "negative capacity field"
    if totals and frees:
        assert max(frees) <= max(totals), "free > total (impossible)"

    # 3. usage broken down by top-level directory
    print("[report] usage by top-level directory:")
    top = [p for p in fs.ls("/") if fs.info(p)["type"] == "directory"]
    grand_total = file_count = 0
    for d in sorted(top):
        size = fs.du(d, total=True)  # total=True -> int; default returns a dict
        n = len(fs.find(d))
        grand_total += size
        file_count += n
        print(f"         {d:<24} {human(size):>10}  ({n} files)")
    print(
        f"[report] total: {human(grand_total)} across {file_count} files "
        f"in {len(top)} top-level dirs"
    )

    # 4. walk() should agree with find() on the count
    walked = sum(len(files) for _, _, files in fs.walk("/"))
    assert walked == file_count, f"walk() {walked} != find() {file_count}"
    print(f"[report] walk() counted {walked} files (agrees with find)")


    s = fs.metrics.summary()
    print(
        f"[report] {s['total_operations']} operations, {s['total_errors']} errors, "
        f"{s['uptime_seconds']:.2f}s"
    )
    assert s["total_errors"] == 0

    fs.close()
    print("[report] OK: capacity/usage dashboard ran clean against real DAOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
