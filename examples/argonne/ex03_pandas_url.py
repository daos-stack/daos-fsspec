#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex03 - pandas via a dfs:// URL
==============================
The point of fsspec: existing application code needs no changes. pandas accepts
a dfs:// URL wherever it accepts a path; no filesystem object to pass around,
no open(), no staging to local disk."""

import os

import fsspec
import pandas as pd


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex03_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

# The URL form. Anything that accepts a path in pandas accepts this.
URL = f"dfs://{POOL}/{CONT}{RDIR}/people.csv"

df = pd.DataFrame({"name": ["ada", "alan", "grace"], "n": [1, 2, 3]})
df.to_csv(URL, index=False)
log("wrote         :", URL)

back = pd.read_csv(URL)
log("read back     :", back.to_dict("records"))
log("columns       :", list(back.columns))
log("rows          :", len(back))
log("roundtrip eq  :", df.equals(back))

# parquet works the same way
purl = f"dfs://{POOL}/{CONT}{RDIR}/people.parquet"
df.to_parquet(purl)
log("parquet rows  :", len(pd.read_parquet(purl)))

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
