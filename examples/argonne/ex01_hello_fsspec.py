#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex01 - hello daos-fsspec
========================
The smallest useful daos-fsspec program. Connect to a DAOS container, then
write, read, stat, list and delete through the ordinary fsspec API; the same
calls you would make against s3, gcs or a local disk."""

import os

import fsspec


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))

# Rank id goes in the directory name and in every file name (see above).
RDIR = f"/ex01_rank_{RANK:03d}"
HELLO = f"{RDIR}/hello_r{RANK:03d}.txt"
STREAM = f"{RDIR}/stream_r{RANK:03d}.txt"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


# ------------------------------------------------------------------ connect --
# "dfs" is the protocol registered by daos-fsspec. From here on this object
# behaves like any other fsspec filesystem (s3, gcs, local, ...).
fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)
log("workdir       :", RDIR)


# -------------------------------------------------------------------- write --
# pipe_file() writes bytes in one shot. For streaming use fs.open(path, "wb").
fs.pipe_file(HELLO, (f"hello daos-fsspec from rank {RANK:d}").encode())
log("wrote         :", HELLO)


# --------------------------------------------------------------------- read --
# cat_file() reads the whole object. For streaming use fs.open(path, "rb").
log("read back     :", fs.cat_file(HELLO).decode())


# --------------------------------------------------------------------- stat --
log("size          :", fs.info(HELLO)["size"], "bytes")


# ------------------------------------------------------ streaming (bonus) ----
# The same filesystem object works as a file-like source/sink, so libraries
# that take an open file handle (torch.save, h5py, joblib, ...) work unchanged.
with fs.open(STREAM, "wb") as f:
    for i in range(3):
        f.write(b"rank %d line %d\n" % (RANK, i))

with fs.open(STREAM, "r") as f:
    for line in f:
        log("stream line   :", line.rstrip())


# --------------------------------------------------------------------- list --
# Safe: only this rank ever touches RDIR.
log("ls workdir    :", sorted(p.rsplit("/", 1)[-1] for p in fs.ls(RDIR, detail=False)))


# ------------------------------------------------------------------- delete --
fs.rm(RDIR, recursive=True)
log("deleted       :", RDIR, "| exists now:", fs.exists(RDIR))

log("done")
