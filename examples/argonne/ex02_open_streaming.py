#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex02 - streaming open()
=======================
fs.open() gives a file-like object, so data moves in chunks and memory stays
flat regardless of object size. Prefer it over pipe_file/cat_file whenever the
payload is large or arrives incrementally."""

import os

import fsspec


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex02_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

P = f"{RDIR}/stream.bin"

# --- streaming write: one chunk at a time, memory stays flat ----------------
with fs.open(P, "wb") as f:
    for i in range(5):
        f.write(b"line %d\n" % i)
log("wrote         : 5 lines to", P)

# --- partial read, then seek back to the start ------------------------------
with fs.open(P, "rb") as f:
    log("first 7 bytes :", f.read(7))
    f.seek(0)
    text = f.read().decode()
    log("all lines     :", text.split())

# --- text mode gives line iteration -----------------------------------------
with fs.open(P, "r") as f:
    for line in f:
        log("iterated      :", line.rstrip())

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
