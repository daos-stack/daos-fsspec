#!/usr/bin/env python3
"""
Example 2: Crash-safe training checkpoints on DAOS, with resume-from-latest.

Preemptible training jobs dump a checkpoint every epoch to shared storage and,
on restart, pick up from the newest one. The safe pattern is write-to-temp then
atomically rename into place, so a killed process never leaves a half-written
checkpoint behind.

Note: DAOS DFS does not support O_APPEND, so the metrics log is updated with a
read-modify-write instead of an append.

Run:
    export DAOS_POOL=mypool DAOS_CONTAINER=mycont
    python examples/checkpoint_store.py
"""

import os
import sys

import fsspec

POOL = os.environ.get("DAOS_POOL")
CONT = os.environ.get("DAOS_CONTAINER")
if not (POOL and CONT):
    sys.exit("set DAOS_POOL and DAOS_CONTAINER first")

RUN_DIR = "/experiments/run-42"
CKPT_BYTES = 4 * 1024 * 1024  # pretend the model weights are 4 MiB


def fake_weights(epoch: int) -> bytes:
    return (f"epoch={epoch};".encode()) + bytes((epoch * 7 + i) % 256 for i in range(64)) * (
        CKPT_BYTES // 64
    )


def save_checkpoint(fs, epoch: int, loss: float) -> None:
    tmp = f"{RUN_DIR}/checkpoint-{epoch:03d}.pt.tmp"
    final = f"{RUN_DIR}/checkpoint-{epoch:03d}.pt"

    with fs.open(tmp, "wb") as f:  # write to a temp path...
        f.write(fake_weights(epoch))
    fs.mv(tmp, final)  # ...then atomically promote it (crash-safe)

    # append a metrics row via read-modify-write (no O_APPEND on DFS)
    metrics = f"{RUN_DIR}/metrics.csv"
    prev = fs.cat_file(metrics).decode() if fs.exists(metrics) else "epoch,loss\n"
    fs.pipe_file(metrics, (prev + f"{epoch},{loss:.4f}\n").encode())


def latest_epoch(fs) -> int:
    ckpts = [p.split("/")[-1] for p in fs.ls(RUN_DIR) if p.endswith(".pt")]
    return max((int(c.split("-")[1].split(".")[0]) for c in ckpts), default=-1)


def main() -> int:
    fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)

    if fs.exists(RUN_DIR):
        fs.rm(RUN_DIR, recursive=True)
    fs.mkdir(RUN_DIR, create_parents=True)
    print(f"[checkpoint] run dir -> {RUN_DIR}")

    # ---- job A: train 5 epochs, then get preempted ----
    for epoch in range(5):
        save_checkpoint(fs, epoch, 1.0 / (epoch + 1))
    print(f"[checkpoint] job A wrote checkpoints 0..4, latest={latest_epoch(fs)}")
    assert latest_epoch(fs) == 4
    assert not [p for p in fs.ls(RUN_DIR) if p.endswith(".tmp")], "temp file leaked"

    # ---- job B: resume from latest, verify content, train 3 more ----
    resume = latest_epoch(fs)
    blob = fs.cat_file(f"{RUN_DIR}/checkpoint-{resume:03d}.pt")
    assert blob == fake_weights(resume), "resumed checkpoint content mismatch!"
    print(f"[checkpoint] job B resumed from epoch {resume}, verified ({len(blob):,} B)")

    for epoch in range(resume + 1, resume + 4):
        save_checkpoint(fs, epoch, 1.0 / (epoch + 1))
    assert latest_epoch(fs) == 7
    print(f"[checkpoint] job B trained through epoch {latest_epoch(fs)}")

    # ---- metrics log accumulated all 8 rows ----
    rows = fs.cat_file(f"{RUN_DIR}/metrics.csv").decode().strip().splitlines()
    assert len(rows) == 1 + 8 and rows[0] == "epoch,loss"
    print(f"[checkpoint] metrics.csv has {len(rows) - 1} rows, header intact")

    # du() returns a per-file dict by default; pass total=True for an int
    total = fs.du(RUN_DIR, total=True)
    info = fs.info(f"{RUN_DIR}/checkpoint-007.pt")
    print(
        f"[checkpoint] run size = {total / 1e6:.1f} MB across "
        f"{len(fs.ls(RUN_DIR))} files; newest mode={oct(info['mode'])}"
    )

    print("[checkpoint] OK: crash-safe checkpoint + resume works on DAOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
