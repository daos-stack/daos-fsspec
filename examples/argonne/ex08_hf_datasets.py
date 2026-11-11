#!/usr/bin/env python3
# Originally written by Kaushik Velusamy, Argonne Leadership Computing Facility.

"""ex08 - HuggingFace datasets
===========================
save_to_disk() to a dfs:// URL works on stock daos-fsspec."""

import os
import tempfile

import fsspec
from datasets import Dataset, load_from_disk


POOL = os.environ.get("DAOS_POOL", "datascience")
CONT = os.environ.get("DAOS_CONT", "fsspec_bench_dfs")


RANK = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", 1)))
RDIR = f"/ex08_rank_{RANK:03d}"


def log(*a):
    print(f"[rank {RANK:03d}]", *a, flush=True)


os.environ["HF_HUB_OFFLINE"] = "1"  # Aurora compute nodes have no internet

fs = fsspec.filesystem("dfs", pool=POOL, container=CONT)
log(f"connected to  : dfs://{POOL}/{CONT}  (rank {RANK:d} of {SIZE:d})")


if fs.exists(RDIR):
    fs.rm(RDIR, recursive=True)
fs.mkdirs(RDIR)

URI = f"dfs://{POOL}/{CONT}{RDIR}/ds"
D = f"{RDIR}/ds"
SO = {"pool": POOL, "container": CONT}


ds = Dataset.from_dict(
    {
        "text": [f"sample sentence number {i:d}" for i in range(500)],
        "label": [i % 3 for i in range(500)],
    }
)
ds.save_to_disk(URI, storage_options=SO)
log("saved to      :", URI)
log("objects writtn:", len(fs.find(D)))

with tempfile.TemporaryDirectory() as td:

    local = os.path.join(td, "ds")
    fs.get(D, local, recursive=True)
    back = load_from_disk(local)
    log("rows          :", back.num_rows)
    log("columns       :", back.column_names)
    log("row 0         :", back[0])
    log("roundtrip ok  :", back[:]["text"] == ds[:]["text"])

    # arrow cannot mmap a remote FS, so datasets streams instead - map() works
    doubled = back.map(lambda r: {"label2": r["label"] * 2})
    log("after map     :", doubled.column_names)

fs.rm(RDIR, recursive=True)
log("cleaned up    :", RDIR, "| exists now:", fs.exists(RDIR))
log("done")
