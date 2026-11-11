#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex12 - Arrow / parquet
======================
pyarrow speaks fsspec natively: pass filesystem=fs and everything works --
column projection, metadata-only reads, hive partitioning and predicate
pushdown."""

import os

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex12_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

P, PARTS = f"{RDIR}/big.parquet", f"{RDIR}/parts"

n = 50_000
tbl = pa.table(
    {"id": np.arange(n), "value": np.random.rand(n), "group": np.random.choice(["a", "b", "c"], n)}
)

# --- single file --------------------------------------------------------------
pq.write_table(tbl, P, filesystem=fs, compression="snappy")
log("wrote parquet :", fs.info(P)["size"], "bytes for", n, "rows")

back = pq.read_table(P, filesystem=fs)
log("rows read     :", back.num_rows, "| equal:", back.equals(tbl))

# --- column projection: read 1 of 3 columns off disk -------------------------
one = pq.read_table(P, columns=["value"], filesystem=fs)
log("1-col read    :", one.num_columns, "column,", one.num_rows, "rows")

# --- metadata only, no data read ---------------------------------------------
md = pq.read_metadata(P, filesystem=fs)
log("metadata only : rows =", md.num_rows, "row_groups =", md.num_row_groups)

# --- hive-partitioned dataset + predicate pushdown ---------------------------
ds.write_dataset(
    tbl,
    PARTS,
    format="parquet",
    filesystem=fs,
    partitioning=ds.partitioning(pa.schema([("group", pa.string())]), flavor="hive"),
    existing_data_behavior="overwrite_or_ignore",
)
log("partitions    :", sorted(p.rsplit("/", 1)[-1] for p in fs.ls(PARTS, detail=False)))

d = ds.dataset(PARTS, format="parquet", filesystem=fs, partitioning="hive")
sub = d.to_table(filter=ds.field("group") == "b")
log("filtered rows :", sub.num_rows, "of", n)
log("only group b  :", set(sub.column("group").to_pylist()) == {"b"})

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
