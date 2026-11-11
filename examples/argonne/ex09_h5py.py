#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex09 - HDF5 (h5py)
==================
Two routes are shown, and **both work**. Handing h5py the fsspec file object
directly is the tidier one and is used for the streaming read; building the
file in memory and writing it as a single object is kept because that is the
right shape for writes, where HDF5 wants many small seeks that are far cheaper
in memory than as round trips."""

import io
import os

import fsspec
import h5py
import numpy as np


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex09_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

P = f"{RDIR}/sim.h5"
data = np.arange(1000, dtype="float64").reshape(100, 10)

# --- write: build in memory, then one object write ---------------------------
buf = io.BytesIO()
with h5py.File(buf, "w") as h5:
    h5.create_dataset("temps", data=data, compression="gzip")
    h5.attrs["units"] = "kelvin"
fs.pipe_file(P, buf.getvalue())
log("wrote sim.h5  :", fs.info(P)["size"], "bytes (gzip)")

# --- read: hand h5py the fsspec file object directly -------------------------
# No try/except and no BytesIO round trip. DFSFile is an
# AbstractBufferedFile, which is seekable, so h5py drives it the same way it
# drives a local file; and it reads only the chunks it needs rather than
# pulling the whole object into memory first.
with fs.open(P, "rb") as f, h5py.File(f, "r") as h5:
    log("datasets      :", list(h5))
    log("attrs         :", dict(h5.attrs))
    back = h5["temps"][:]
    log("shape         :", back.shape)
    log("slice [0,:3]  :", back[0, :3])
    log("equal         :", np.array_equal(back, data))

# --- read: whole-object route, for comparison --------------------------------
# Still useful when you want exactly one round trip and the file fits in memory.
with h5py.File(io.BytesIO(fs.cat_file(P)), "r") as h5:
    log("via cat_file  :", h5["temps"].shape)

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
