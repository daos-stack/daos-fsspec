#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex13 - chunked array store (zarr-style)
=======================================
zarr is not installed in the Aurora frameworks module, but its central pattern
-- one object per chunk, read only the chunks you need; is a few lines
directly on fsspec, and it suits an object store exactly."""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex13_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

full = np.arange(400, dtype="float32").reshape(20, 20)
CH = 5  # 5x5 chunks -> 16 chunks


def chunk_path(i, j):
    return f"{RDIR}/c_{i:d}_{j:d}.npy"


def read_chunk(i, j):
    return np.load(io.BytesIO(fs.cat_file(chunk_path(i, j))))


for i in range(0, 20, CH):
    for j in range(0, 20, CH):
        buf = io.BytesIO()
        np.save(buf, full[i : i + CH, j : j + CH])
        fs.pipe_file(chunk_path(i, j), buf.getvalue())
log("wrote chunks  :", len(fs.find(RDIR)), f"of {CH:d}x{CH:d}")

# --- read ONE chunk: the other 15 are never touched --------------------------
c = read_chunk(10, 5)
log("chunk (10,5)  :", c.shape, "first row", c[0])
log("matches source:", np.array_equal(c, full[10:15, 5:10]))

# --- reassemble everything ---------------------------------------------------
out = np.zeros_like(full)
for i in range(0, 20, CH):
    for j in range(0, 20, CH):
        out[i : i + CH, j : j + CH] = read_chunk(i, j)
log("reassembly eq :", np.array_equal(out, full))

# --- concurrent chunk reads: independent objects, so this scales -------------
coords = [(i, j) for i in range(0, 20, CH) for j in range(0, 20, CH)]
with ThreadPoolExecutor(max_workers=4) as ex:
    chunks = list(ex.map(lambda c: read_chunk(*c), coords))
log("threaded read :", len(chunks), "chunks OK")

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
