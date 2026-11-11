#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex21 - SQLite and the stage-local pattern
=========================================
SQLite needs byte-range locking and mmap, neither of which an object store can
offer, so a database cannot be opened in place over fsspec. The correct pattern
-- and the one for any tool that demands a real POSIX file; is:"""

import os
import sqlite3
import tempfile

import fsspec


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex21_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

REMOTE = f"{RDIR}/runs.db"

# --- build locally. Per-rank scratch dir so 12 ranks never collide. ----------
scratch = tempfile.mkdtemp(prefix=f"ex21_r{RANK:03d}_")
local = os.path.join(scratch, "runs.db")

con = sqlite3.connect(local)
con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, model TEXT, acc REAL)")
con.executemany(
    "INSERT INTO runs (model, acc) VALUES (?, ?)",
    [("resnet50", 0.76), ("vit_b", 0.81), ("convnext", 0.84)],
)
con.commit()
con.close()
log("built local db:", os.path.getsize(local), "bytes at", local)

# --- upload as one object -----------------------------------------------------
with open(local, "rb") as f:
    fs.pipe_file(REMOTE, f.read())
log("uploaded      :", fs.info(REMOTE)["size"], "bytes")

# --- download, then query ------------------------------------------------------
local2 = os.path.join(tempfile.mkdtemp(prefix=f"ex21_r{RANK:03d}_"), "runs.db")
fs.get_file(REMOTE, local2)
con = sqlite3.connect(local2)
rows = con.execute("SELECT model, acc FROM runs ORDER BY acc DESC").fetchall()
con.close()
log("query result  :", rows)
log("best model    :", rows[0][0], rows[0][1])

# --- opening in place is NOT supported - this is the documented limitation ---
try:
    con = sqlite3.connect(f"dfs://{POOL}/{CONT}{REMOTE}")
    con.execute("SELECT 1").fetchone()
    log("in-place open : unexpectedly worked")
    con.close()
except Exception as e:
    log("in-place open : fails as expected ->", type(e).__name__)

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
