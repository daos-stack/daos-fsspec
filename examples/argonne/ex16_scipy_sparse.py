#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex16 - scipy sparse + .mat
==========================
Sparse matrices are the natural format for graphs, recommender systems and FEM
work. The two scipy entry points want different things: sparse.save_npz takes a
file object, while scipy.io.savemat works in bytes."""

import io
import os

import fsspec
import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex16_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

SP, MAT = f"{RDIR}/sparse.npz", f"{RDIR}/data.mat"

# --- a 10k x 10k sparse matrix, 0.1% dense -----------------------------------
n = 10_000
rng = np.random.default_rng(0)
m = sparse.csr_matrix(
    (rng.random(100_000), (rng.integers(0, n, 100_000), rng.integers(0, n, 100_000))), shape=(n, n)
)
log("sparse matrix :", m.shape, "nnz =", m.nnz, f"(dense would be {n * n * 8 / 1e6:.0f} MB)")

with fs.open(SP, "wb") as f:
    sparse.save_npz(f, m)
log("saved sparse  :", fs.info(SP)["size"], "bytes")

with fs.open(SP, "rb") as f:
    back = sparse.load_npz(f)
log("nnz preserved :", back.nnz == m.nnz, "| shape", back.shape)
log("values equal  :", np.allclose(back.toarray()[:50, :50], m.toarray()[:50, :50]))

# --- MATLAB .mat interchange --------------------------------------------------
buf = io.BytesIO()
savemat(buf, {"grid": np.arange(12).reshape(3, 4), "label": "run42"})
fs.pipe_file(MAT, buf.getvalue())
md = loadmat(io.BytesIO(fs.cat_file(MAT)))
log("mat keys      :", [k for k in md if not k.startswith("__")])
log("mat grid[0]   :", list(md["grid"][0]))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
