#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex15 - streaming large files
============================
Reading a 500 MB object with cat_file() means 500 MB resident. These patterns
keep memory flat whatever the file size, which matters when twelve ranks share
one node."""

import os

import fsspec
import numpy as np


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex15_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

P = f"{RDIR}/stream.bin"
CHUNK, NCHUNK = 1 << 20, 32  # 1 MB x 32 = 32 MB

# --- build 32 MB by streaming, never holding it whole ------------------------
with fs.open(P, "wb") as f:
    for i in range(NCHUNK):
        f.write(bytes([i % 256]) * CHUNK)
size = fs.info(P)["size"]
log("wrote         :", size // (1 << 20), "MB in", NCHUNK, "streamed chunks")

# --- read back in chunks, memory stays flat ----------------------------------
total, seen = 0, []
with fs.open(P, "rb") as f:
    while True:
        b = f.read(CHUNK)
        if not b:
            break
        total += len(b)
        seen.append(b[0])
log("streamed read :", total // (1 << 20), "MB in", len(seen), "chunks")
log("content order :", seen[:5], "...")

# --- random access: seek straight to an offset -------------------------------
with fs.open(P, "rb") as f:
    f.seek(20 * CHUNK)
    b = f.read(16)
log("seek to 20MB  : byte value", b[0], "(expected 20)")

# --- byte-range read with no file object at all ------------------------------
tail = fs.cat_file(P, start=size - 8)
log("last 8 bytes  : value", tail[0], "len", len(tail))

arr = np.frombuffer(fs.cat_file(P, start=0, end=1024), dtype="uint8")
log("first 1KB     :", arr.shape, "first value", arr[0])

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
