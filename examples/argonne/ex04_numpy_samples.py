#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex04 - numpy .npz samples
=========================
One .npz object per sample, the classic training-data layout on an object
store. numpy serialises to bytes rather than to a path, so the pattern is
BytesIO then pipe_file() on the way out, and cat_file() then BytesIO back."""

import io
import os

import fsspec
import numpy as np


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex04_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

# --- write 5 samples ---------------------------------------------------------
for i in range(5):
    buf = io.BytesIO()
    np.savez(buf, x=np.random.rand(4, 4))
    fs.pipe_file(f"{RDIR}/sample_{i:d}.npz", buf.getvalue())
log("wrote         : 5 .npz samples")

# find() returns FILES only (no directories) - the right call for a file list
files = sorted(fs.find(RDIR))
log("found         :", [f.rsplit("/", 1)[-1] for f in files])

# --- read one back -----------------------------------------------------------
arr = np.load(io.BytesIO(fs.cat_file(files[0])))["x"]
log("sample 0 shape:", arr.shape)
log("sample 0 mean :", round(float(arr.mean()), 4))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
